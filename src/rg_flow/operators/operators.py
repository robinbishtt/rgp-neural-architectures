from __future__ import annotations
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
class StandardRGOperator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        sigma_w: float = 1.4,
        sigma_b: float = 0.3,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_fn = self._get_activation(activation)
        self._init_critical(sigma_w, sigma_b)
    def _get_activation(self, name: str) -> Callable:
        return {"tanh": torch.tanh, "relu": F.relu, "gelu": F.gelu}.get(name, torch.tanh)
    def _init_critical(self, sigma_w: float, sigma_b: float) -> None:
        n = self.linear.weight.shape[1]
        nn.init.normal_(self.linear.weight, std=sigma_w / math.sqrt(n))
        nn.init.normal_(self.linear.bias,   std=sigma_b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.linear(x))
class ResidualRGOperator(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str = "tanh") -> None:
        super().__init__()
        self.op = StandardRGOperator(in_features, out_features, activation)
        self.proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features else nn.Identity()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x) + self.proj(x)
class AttentionRGOperator(nn.Module):
    def __init__(self, features: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(features, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(features)
        self.ff   = nn.Sequential(
            nn.Linear(features, features * 2),
            nn.GELU(),
            nn.Linear(features * 2, features),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.norm(x + self.ff(x)).squeeze(1)
class WaveletRGOperator(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        assert features % 2 == 0, "features must be even for Haar decomposition"
        self.low_pass  = nn.Linear(features // 2, features)
        self.high_pass = nn.Linear(features // 2, features)
        self.combine   = nn.Linear(features * 2, features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        lo = (x[..., :half] + x[..., half:]) / math.sqrt(2.0)
        hi = (x[..., :half] - x[..., half:]) / math.sqrt(2.0)
        out = torch.cat([self.low_pass(lo), self.high_pass(hi)], dim=-1)
        return torch.tanh(self.combine(out))
class LearnedRGOperator(nn.Module):
    def __init__(self, features: int, context_dim: int = 16) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(features, context_dim),
            nn.Tanh(),
        )
        self.scale_net = nn.Linear(context_dim, features)
        self.shift_net = nn.Linear(context_dim, features)
        self.base_op   = StandardRGOperator(features, features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx   = self.context_encoder(x.detach())
        scale = torch.sigmoid(self.scale_net(ctx))
        shift = self.shift_net(ctx)
        return scale * self.base_op(x) + shift