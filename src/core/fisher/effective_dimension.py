"""
src/core/fisher/effective_dimension.py

Effective dimensionality of Fisher information matrices.

Three definitions implemented:
  1. Participation Ratio (PR)         d_eff = (Σλ)² / Σλ²
  2. Spectral Entropy                 H = -Σ p_i log p_i
  3. Rank Threshold                   r_ε = |{i : λ_i > ε}|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EffectiveDimensionResult:
    participation_ratio: float
    spectral_entropy:    float
    rank_threshold:      int
    n_total:             int
    compression_ratio:   float


class FisherEffectiveDimension:
    """
    Computes effective dimensionality of Fisher information geometry.

    The participation ratio d_eff = (Σλ_i)² / Σλ_i² measures how many
    eigenvalue directions carry significant information. For a uniform
    spectrum d_eff = N; for a rank-1 matrix d_eff = 1.
    """

    def __init__(self, rank_threshold_eps: float = 1e-3) -> None:
        self.eps = rank_threshold_eps

    def compute(self, G: torch.Tensor) -> EffectiveDimensionResult:
        """
        Full effective dimension analysis of Fisher matrix G.

        Parameters
        ----------
        G : (N, N) Fisher information matrix

        Returns
        -------
        EffectiveDimensionResult with multiple definitions
        """
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        n  = len(ev)

        # Participation ratio
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())

        # Spectral entropy
        p_norm  = ev / ev.sum()
        entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

        # Rank threshold: eigenvalues above eps * max
        threshold = self.eps * ev.max()
        rank_t    = int((ev > threshold).sum())

        return EffectiveDimensionResult(
            participation_ratio=pr,
            spectral_entropy=entropy,
            rank_threshold=rank_t,
            n_total=n,
            compression_ratio=float(pr / n),
        )

    def layer_profile(
        self, fisher_matrices: list
    ) -> list:
        """Compute effective dimension at each layer."""
        return [self.compute(G) for G in fisher_matrices]
 