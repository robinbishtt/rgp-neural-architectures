"""
src/architectures/baselines/transformer_baseline.py

TransformerBaseline: standard transformer encoder (without positional
embeddings) for baseline comparison against RG-Net architectures.
Used in H3 multi-scale generalization experiments.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerBaseline(nn.Module):
    """
    Transformer encoder baseline for benchmark comparison.

    Input features are treated as a sequence of tokens; no positional
    encoding is applied so the model is permutation-invariant (matching
    the symmetry of feedforward baselines). Classification is performed
    via mean-pooling of the final layer's token representations.
    """

    def __init__(
        self,
        input_dim:  int,
        n_classes:  int,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        d_ff:       int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder    = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim) → treat as single-token sequence
        x = self.input_proj(x)
        for block in self.encoder:
            x = block(x)
        x = x.mean(dim=1)       # mean pool over sequence
        return self.classifier(x)
 