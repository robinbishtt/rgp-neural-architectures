"""
src/architectures/baselines/attention_baseline.py

AttentionBaseline: lightweight self-attention network (no positional
encoding, no FF sub-layers) for ablation comparison. Tests whether
the RG-Net performance advantage comes specifically from the RG-inspired
depth structure or is reproduced by any attention mechanism.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class AttentionBaseline(nn.Module):
    """
    Stacked self-attention baseline without feedforward sub-layers.

    Provides a controlled comparison to TransformerBaseline (which includes
    feedforward layers) and to RG-Net (which uses structured RG coarse-graining).
    The architecture intentionally lacks the FF component to isolate the
    contribution of attention-based feature mixing.
    """

    def __init__(
        self,
        input_dim:  int,
        n_classes:  int,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 6,
    ) -> None:
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.layers      = nn.ModuleList([
            SelfAttentionLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
 