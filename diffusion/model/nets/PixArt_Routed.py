import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import (
    t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention,
    T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
)
from diffusion.utils.logger import get_root_logger
from diffusion.model.nets.routed_ffn import RoutedFFN


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.,
        window_size=0,
        input_size=None,
        use_rel_pos=False,
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos,
            **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = RoutedFFN(
            hidden_size=hidden_size,
            light_ratio=2.0,
            heavy_ratio=mlp_ratio,
            act_layer=approx_gelu,
            drop=0.0,
        )

        self.last_router_loss = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        attn_in = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.drop_path(gate_msa * self.attn(attn_in).reshape(B, N, C))

        x = x + self.cross_attn(x, y, mask)

        mlp_in = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mlp_in)
        self.last_router_loss = getattr(self.mlp, "last_router_loss", None)

        x = x + self.drop_path(gate_mlp * mlp_out)
        return x


class PixArt(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.,
        window_size=0,
        window_block_indexes=None,
        use_rel_pos=False,
        caption_channels=4096,
        lewei_scale=1.0,
        config=None,
        model_max_length=120,
        **kwargs
    ):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                window_size=window_size if i in window_block_indexes else 0,
                use_rel_pos=use_rel_pos if i in window_block_indexes else False,
            )
            for i in range(depth)
        ])

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        self.router_loss = 0.0
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep.to(x.dtype))
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        self.router_loss = 0.0
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)
            block_router_loss = getattr(block, "last_router_loss", None)
            if block_router_loss is not None:
                self.router_loss += block_router_loss

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
            lewei_scale=self.lewei_scale,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


@MODELS.register_module()
def PixArt_XL_2_Routed(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)