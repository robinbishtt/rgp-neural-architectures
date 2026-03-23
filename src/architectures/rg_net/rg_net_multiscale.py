"""
src/architectures/rg_net/rg_net_multiscale.py

RGNetMultiScale - explicit multi-scale feature fusion.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


class RGNetMultiScale(nn.Module):
    """
    Multi-scale RG-Net with explicit feature fusion across depth levels.

    Extracts intermediate representations at depths [L/4, L/2, 3L/4, L]
    and fuses them for the final classification head.
    """

    def __init__(
        self,
        in_features: int = 784,
        n_classes: int = 10,
        depth: int = 100,
        width: int = 256,
        fusion_points: List[float] = (0.25, 0.5, 0.75, 1.0),
    ) -> None:
        super().__init__()
        self.fusion_depths = [max(1, int(depth * f)) for f in fusion_points]
        self.depth         = depth
        self.input_proj    = nn.Linear(in_features, width)
        self.layers        = nn.ModuleList(
            [StandardRGOperator(width, width) for _ in range(depth)]
        )
        # Fusion projections: each scale -> fusion_dim
        fusion_dim = width // len(fusion_points)
        self.fusion_projs  = nn.ModuleList(
            [nn.Linear(width, fusion_dim) for _ in fusion_points]
        )
        self.head = nn.Linear(fusion_dim * len(fusion_points), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        h = F.tanh(self.input_proj(x))
        scale_feats = []
        next_fusion = 0

        for k, layer in enumerate(self.layers):
            h = layer(h)
            if next_fusion < len(self.fusion_depths) and k + 1 == self.fusion_depths[next_fusion]:
                proj = self.fusion_projs[next_fusion](h)
                scale_feats.append(proj)
                next_fusion += 1

        while len(scale_feats) < len(self.fusion_projs):
            scale_feats.append(self.fusion_projs[len(scale_feats)](h))

        return self.head(torch.cat(scale_feats, dim=-1))
 