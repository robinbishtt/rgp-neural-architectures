"""
src/rg_flow/operators/attention_rg_operator.py

AttentionRGOperator: attention-based RG coarse-graining for capturing
long-range dependencies across scales. Implements the multi-head self-
attention RG transformation where the coarse-graining kernel is learned
from data rather than fixed.

Physical interpretation: classical RG coarse-graining replaces groups of
fine-grained degrees of freedom with a single representative. Attention-
based coarse-graining learns which fine-grained features to aggregate
for each coarse-grained unit, enabling adaptive scale selection.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionRGOperator(nn.Module):
    """
    Attention-based RG transformation for long-range coarse-graining.

    Architecture:
        h^(k) = Attention(Q=h^(k-1), K=h^(k-1), V=h^(k-1)) W_O
    where attention weights define the coarse-graining kernel.

    The multi-head formulation decomposes the feature space into n_heads
    independent subspaces, each performing its own RG coarse-graining.
    Cross-head aggregation then combines the multi-scale representations.

    Critical initialization: Q, K weight matrices initialized with
    σ_w² = 1/d_head to ensure unit-norm attention logits at initialization,
    preserving the edge-of-chaos property in the attention subspace.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int   = 4,
        dropout:     float = 0.0,
        bias:        bool  = True,
    ) -> None:
        """
        Args:
            d_model: feature dimension (must be divisible by n_heads)
            n_heads: number of attention heads
            dropout: attention dropout probability
            bias:    whether to use bias in projection layers
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Critical initialization: σ_w = 1/sqrt(d_head) for Q, K
        nn.init.normal_(self.W_q.weight, std=1.0 / self.d_head ** 0.5)
        nn.init.normal_(self.W_k.weight, std=1.0 / self.d_head ** 0.5)
        nn.init.normal_(self.W_v.weight, std=1.0 / d_model ** 0.5)
        nn.init.zeros_(self.W_o.weight)
        if bias:
            nn.init.zeros_(self.W_q.bias)
            nn.init.zeros_(self.W_k.bias)
            nn.init.zeros_(self.W_v.bias)
            nn.init.zeros_(self.W_o.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based RG coarse-graining.

        Args:
            x: (..., d_model) input feature tensor

        Returns:
            h: (..., d_model) coarse-grained representation
        """
        *batch, d = x.shape
        # Reshape for multi-head attention if 3D (batch, seq, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, d_model)

        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)
        out    = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out    = self.W_o(out)

        # Residual connection + squeeze if 1 token
        out = x + out
        if len(batch) == 1:
            out = out.squeeze(1)  # (B, d_model)
        return out
