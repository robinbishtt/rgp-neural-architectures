"""
src/core/fisher/eigenvalue_analyzer.py

Spectral decomposition and eigenvalue density estimation of Fisher information matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class EigenvalueAnalysisResult:
    eigenvalues:          np.ndarray
    effective_dimension:  float
    condition_number:     float
    participation_ratio:  float
    spectral_entropy:     float


class FisherEigenvalueAnalyzer:
    """
    Full spectral analysis of Fisher information matrices.

    Computes: eigenvalue spectrum, effective dimension, condition number,
    participation ratio (IPR), and spectral entropy.
    """

    def analyze(self, G: torch.Tensor) -> EigenvalueAnalysisResult:
        """
        Full spectral analysis of Fisher matrix G.

        Parameters
        ----------
        G : (N, N) symmetric positive semi-definite tensor

        Returns
        -------
        EigenvalueAnalysisResult with all spectral quantities
        """
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)

        d_eff   = float(ev.sum() ** 2 / (ev ** 2).sum())
        kappa   = float(ev[-1] / ev[0])
        pr      = float(ev.sum() ** 2 / ((ev ** 2).sum() * len(ev)))
        p_norm  = ev / ev.sum()
        entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

        return EigenvalueAnalysisResult(
            eigenvalues=ev,
            effective_dimension=d_eff,
            condition_number=kappa,
            participation_ratio=pr,
            spectral_entropy=entropy,
        )

    def dominant_subspace(
        self, G: torch.Tensor, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-n eigenvalues and eigenvectors."""
        ev, V = torch.linalg.eigh(G)
        ev = ev.cpu().numpy()
        V  = V.cpu().numpy()
        idx = np.argsort(ev)[::-1]
        return ev[idx[:n_components]], V[:, idx[:n_components]]

    def variance_explained(self, G: torch.Tensor, n_components: int) -> float:
        """Fraction of total variance captured by top-n components."""
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        top = np.sort(ev)[::-1][:n_components]
        return float(top.sum() / ev.sum())
