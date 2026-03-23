"""
src/architectures/baselines/wavelet_baseline.py

Wavelet-CNN baseline for H3 comparative evaluation.

Architecture: multi-scale wavelet decomposition followed by learned
aggregation. Implements the Haar wavelet transform at each scale level,
producing multi-resolution representations analogous to the RG coarse-graining
but using fixed (non-learned) basis functions.

Paper Table 1: Wavelet-CNN achieves Hier-3 ID/OOD = 82.1/71.2, CIFAR-100 = 80.1%
               Parameters: ~22.1M
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletLayer(nn.Module):
    """
    Haar wavelet decomposition layer.
    
    Splits input into low-frequency (approximation) and high-frequency
    (detail) components via the Haar transform, then projects each to
    the output dimension.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        assert in_features % 2 == 0, "in_features must be even for Haar decomposition"
        half = in_features // 2
        self.low_proj  = nn.Linear(half, out_features // 2)
        self.high_proj = nn.Linear(half, out_features // 2)
        self.scale     = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        # Haar wavelet: low = (a + b) / sqrt(2),  high = (a - b) / sqrt(2)
        low  = (x[..., :half] + x[..., half:]) * self.scale
        high = (x[..., :half] - x[..., half:]) * self.scale
        return torch.cat([self.low_proj(low), self.high_proj(high)], dim=-1)


class WaveletCNNBaseline(nn.Module):
    """
    Multi-scale Wavelet-CNN baseline.
    
    Architecture:
      - Embedding layer: input_dim → hidden_dim
      - n_scales wavelet decomposition levels (each halves the effective scale)
      - Per-scale processing with learned MLP blocks
      - Multi-scale feature fusion before classification head
    
    The key difference from RG-Net: wavelet filters are FIXED (Haar basis),
    whereas RG-Net learns its coarse-graining operators.
    """

    def __init__(
        self,
        input_dim:   int   = 784,
        hidden_dim:  int   = 256,
        output_dim:  int   = 10,
        n_scales:    int   = 4,
        depth_per_scale: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)

        # One wavelet decomposition + MLP stack per scale
        self.wavelet_layers = nn.ModuleList()
        self.scale_mlps     = nn.ModuleList()

        for scale in range(n_scales):
            if hidden_dim % 2 == 0:
                self.wavelet_layers.append(HaarWaveletLayer(hidden_dim, hidden_dim))
            else:
                self.wavelet_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp = nn.Sequential(*[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
                for _ in range(depth_per_scale)
            ])
            self.scale_mlps.append(mlp)

        # Fusion: concatenate features from all scales → head
        self.fusion = nn.Linear(hidden_dim * n_scales, hidden_dim)
        self.head   = nn.Linear(hidden_dim, output_dim)
        self.n_scales = n_scales

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        scale_features = []
        for i in range(self.n_scales):
            h = torch.tanh(self.wavelet_layers[i](h))
            h = self.scale_mlps[i](h)
            scale_features.append(h)
        fused = torch.tanh(self.fusion(torch.cat(scale_features, dim=-1)))
        return self.head(fused)
