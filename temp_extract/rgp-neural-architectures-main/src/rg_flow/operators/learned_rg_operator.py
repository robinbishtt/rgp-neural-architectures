"""
src/rg_flow/operators/learned_rg_operator.py

LearnedRGOperator: data-adaptive coarse-graining via meta-learning.
The coarse-graining kernel is not hand-designed but instead learned
from data, allowing the network to discover the optimal renormalization
group transformation for the task at hand.

This implements the most general RG transformation in the framework,
parameterized by a hypernetwork that generates the coarse-graining
weights conditioned on the current representation statistics.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedRGOperator(nn.Module):
    """
    Data-adaptive (learned) RG coarse-graining operator.

    Architecture:
        - HyperNetwork: takes summary statistics of h^(k-1) as input
          and outputs the coarse-graining kernel parameters W_k.
        - CoarseGraining: applies W_k to h^(k-1) to produce h^(k).

    The hyper-network is a small MLP that maps
        φ(h^(k-1)) = [mean, std, skew] → W_k ∈ R^{out_dim × in_dim}
    enabling the coarse-graining weights to adapt to the current
    representation statistics (a form of fast weights / hypernetwork).

    Critical initialization: the hyper-network is initialized to produce
    identity-like transformations at t=0, ensuring the standard
    critical initialization conditions are satisfied at training start.
    """

    def __init__(
        self,
        in_dim:       int,
        out_dim:      int,
        hyper_hidden: int  = 32,
    ) -> None:
        """
        Args:
            in_dim:       input feature dimension
            out_dim:      output feature dimension
            hyper_hidden: hidden units in the hyper-network
        """
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        # HyperNetwork: [mean, std, skew] (3 stats) → weight matrix + bias
        n_stats = 3
        n_params = out_dim * in_dim + out_dim  # W + b
        self.hyper_net = nn.Sequential(
            nn.Linear(n_stats, hyper_hidden),
            nn.Tanh(),
            nn.Linear(hyper_hidden, n_params),
        )

        # Baseline (prior) transformation for residual hyper-network
        self.baseline = nn.Linear(in_dim, out_dim)

        self.activation = nn.Tanh()

        # Critical initialization
        nn.init.normal_(self.baseline.weight, std=1.0 / in_dim ** 0.5)
        nn.init.zeros_(self.baseline.bias)
        # Initialize hyper-network outputs to zero (identity residual)
        nn.init.zeros_(self.hyper_net[-1].weight)
        nn.init.zeros_(self.hyper_net[-1].bias)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute [mean, std, skewness] summary statistics of x."""
        mu     = x.mean(dim=-1, keepdim=True)
        sigma  = x.std(dim=-1, keepdim=True) + 1e-8
        skew   = ((x - mu) ** 3).mean(dim=-1, keepdim=True) / sigma ** 3
        return torch.cat([mu, sigma, skew], dim=-1)  # (B, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned RG coarse-graining.

        Args:
            x: (B, in_dim) input features

        Returns:
            h: (B, out_dim) adaptively coarse-grained representation
        """
        stats  = self._compute_stats(x)           # (B, 3)
        params = self.hyper_net(stats)             # (B, out_dim*in_dim + out_dim)

        W_delta = params[:, :self.out_dim * self.in_dim].view(
            -1, self.out_dim, self.in_dim
        )  # (B, out_dim, in_dim)
        b_delta = params[:, self.out_dim * self.in_dim:]  # (B, out_dim)

        # Baseline transformation
        h_base = self.baseline(x)  # (B, out_dim)

        # Adaptive residual: bmm handles per-sample weight matrices
        h_adapt = torch.bmm(W_delta, x.unsqueeze(-1)).squeeze(-1) + b_delta

        return self.activation(h_base + h_adapt)
