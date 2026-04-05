import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

class RoutedFFN(nn.Module):
    def __init__(self, hidden_size, light_ratio, heavy_ratio, act_layer, drop):
        super().__init__()
        self.router = nn.Linear(hidden_size, 2)

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

    def forward(self, x):

        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)

        out_light = self.light(x)
        out_heavy = self.heavy(x)
        out = probs[..., 0:1] * out_light + probs[..., 1:2] * out_heavy

        target_ratio = 0.3
        light_ratio_actual = probs[..., 0].mean()
        self.last_router_loss = (light_ratio_actual - target_ratio) ** 2

        return out