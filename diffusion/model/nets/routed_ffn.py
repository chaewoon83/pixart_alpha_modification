import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


class RoutedFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        light_ratio,
        heavy_ratio,
        act_layer,
        drop,
        target_ratio=0.2
    ):
        super().__init__()
        self.router = nn.Linear(hidden_size, 2)
        self.target_ratio = target_ratio

        self.light = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * light_ratio),
            act_layer=act_layer,
            drop=drop
        )

        self.heavy = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * heavy_ratio),
            act_layer=act_layer,
            drop=drop
        )

        self.last_router_loss = None
        self.last_entropy = None
        self.last_light_ratio = None
        self.last_heavy_ratio = None
        self.last_confidence = None
        self.last_hard_light_ratio = None
        self.last_hard_heavy_ratio = None

        self.last_routing_map = None   # Last forward pass token-by-token hard routing
        self.total_light_tokens = 0
        self.total_heavy_tokens = 0
        self.total_tokens = 0
        self.num_forwards = 0

    def reset_stats(self):
        self.last_router_loss = None
        self.last_entropy = None
        self.last_light_ratio = None
        self.last_heavy_ratio = None
        self.last_confidence = None
        self.last_hard_light_ratio = None
        self.last_hard_heavy_ratio = None

        self.last_routing_map = None
        self.total_light_tokens = 0
        self.total_heavy_tokens = 0
        self.total_tokens = 0
        self.num_forwards = 0

    def forward(self, x):
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)

        if self.training:
            out_light = self.light(x)
            out_heavy = self.heavy(x)
            out = probs[..., 0:1] * out_light + probs[..., 1:2] * out_heavy
        else:
            hard_choice = probs.argmax(dim=-1)
            self.last_routing_map = hard_choice.detach().cpu()

            out = torch.zeros_like(x)
            light_mask = (hard_choice == 0)
            heavy_mask = (hard_choice == 1)

            if light_mask.any():
                out[light_mask] = self.light(x[light_mask])
            if heavy_mask.any():
                out[heavy_mask] = self.heavy(x[heavy_mask])

            light_count = (hard_choice == 0).sum().item()
            heavy_count = (hard_choice == 1).sum().item()
            total_count = hard_choice.numel()

            self.total_light_tokens += light_count
            self.total_heavy_tokens += heavy_count
            self.total_tokens += total_count
            self.num_forwards += 1

        light_ratio_actual = probs[..., 0].mean()
        heavy_ratio_actual = probs[..., 1].mean()

        router_loss = (light_ratio_actual - self.target_ratio) ** 2
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        confidence = probs.max(dim=-1).values.mean()

        hard_choice = probs.argmax(dim=-1)
        hard_light_ratio = (hard_choice == 0).float().mean()
        hard_heavy_ratio = (hard_choice == 1).float().mean()

        self.last_router_loss = router_loss
        self.last_entropy = entropy
        self.last_light_ratio = light_ratio_actual.detach()
        self.last_heavy_ratio = heavy_ratio_actual.detach()
        self.last_confidence = confidence.detach()
        self.last_hard_light_ratio = hard_light_ratio.detach()
        self.last_hard_heavy_ratio = hard_heavy_ratio.detach()

        return out