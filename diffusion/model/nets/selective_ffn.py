import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


class TokenSelectiveFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        mlp_ratio=4.0,
        target_ratio=0.8,
        lambda_budget=1e-5,
        detach_attn_feat=True,
        gate_temperature=1.0,
        drop=0.0,
        attn_feat_dim=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_ratio = target_ratio
        self.lambda_budget = lambda_budget
        self.detach_attn_feat = detach_attn_feat
        self.gate_temperature = gate_temperature

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=drop,
        )

        self.use_x_for_gate = True

        gate_in_dim = hidden_size + attn_feat_dim if self.use_x_for_gate else attn_feat_dim
        self.router = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )

        self.last_router_loss = None
        self.last_keep_ratio = None
        self.last_hard_keep_ratio = None
        self.last_gate_mean = None
        self.last_selected_ratio = None

        # RoutedFFN-style names for compatibility with existing inference code
        self.last_routing_map = None      # 0 = selected(keep), 1 = skipped
        self.total_light_tokens = 0       # kept tokens
        self.total_heavy_tokens = 0       # skipped tokens
        self.total_tokens = 0
        self.num_forwards = 0

    def reset_stats(self):
        self.last_router_loss = None
        self.last_keep_ratio = None
        self.last_hard_keep_ratio = None
        self.last_gate_mean = None
        self.last_selected_ratio = None

        self.last_routing_map = None
        self.total_light_tokens = 0
        self.total_heavy_tokens = 0
        self.total_tokens = 0
        self.num_forwards = 0

    def forward(self, x, attn_feat):
        if self.detach_attn_feat:
            attn_feat = attn_feat.detach()

        if self.use_x_for_gate:
            gate_in = torch.cat([x, attn_feat], dim=-1)
        else:
            gate_in = attn_feat

        logits = self.router(gate_in).squeeze(-1)           # [B, N]
        probs = torch.sigmoid(logits / self.gate_temperature)  # [B, N]

        keep_ratio = probs.mean()
        self.last_router_loss = self.lambda_budget * (keep_ratio - self.target_ratio) ** 2
        self.last_keep_ratio = keep_ratio.detach()
        self.last_gate_mean = probs.mean().detach()

        hard_mask = (probs > 0.75)  # True = keep FFN, False = skip FFN
        self.last_hard_keep_ratio = hard_mask.float().mean().detach()
        self.last_selected_ratio = self.last_hard_keep_ratio

        # RoutedFFN-style routing map:
        # 0 = light = selected/kept
        # 1 = heavy = skipped
        routing_map = torch.where(
            hard_mask,
            torch.zeros_like(hard_mask, dtype=torch.long),
            torch.ones_like(hard_mask, dtype=torch.long)
        )
        self.last_routing_map = routing_map.detach().cpu()

        if self.training:
            # soft mask training
            mask = probs.unsqueeze(-1)   # [B, N, 1]
            out = self.ffn(x) * mask
            return out

        else:
            # hard mask inference
            out = torch.zeros_like(x)

            selected_x = x[hard_mask]   # [K, C]
            if selected_x.numel() > 0:
                selected_out = self.ffn(selected_x)
                out[hard_mask] = selected_out

            kept_count = hard_mask.sum().item()
            total_count = hard_mask.numel()
            skipped_count = total_count - kept_count

            self.total_light_tokens += kept_count
            self.total_heavy_tokens += skipped_count
            self.total_tokens += total_count
            self.num_forwards += 1

            return out