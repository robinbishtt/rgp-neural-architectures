"""
src/core/spectral/level_spacing.py

Level spacing distribution for universality class identification.

Nearest-neighbor spacing s_i = λ_{i+1} - λ_i (normalized) follows:
  - Poisson:  P(s) = exp(-s)              integrable/uncorrelated
  - GOE:      P(s) ≈ (π/2) s exp(-πs²/4)  orthogonal (Wigner surmise)
  - GUE:      P(s) ≈ (32/π²) s² exp(-4s²/π)  unitary
"""
from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
from scipy.stats import kstest


class LevelSpacingDistribution:
    """
    Nearest-neighbor level spacing statistics for spectral universality.

    Eigenvalue correlations distinguish random (Poisson) from correlated
    (Wigner-Dyson) spectra, identifying the RMT universality class.
    """

    def compute_spacings(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Compute normalized nearest-neighbor spacings."""
        ev = np.sort(np.asarray(eigenvalues, dtype=float))
        s  = np.diff(ev)
        return s / (s.mean() + 1e-12)

    def wigner_surmise_pdf(
        self, s: np.ndarray, ensemble: Literal["GOE", "GUE", "GSE"] = "GOE"
    ) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if ensemble == "GOE":
            return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)
        if ensemble == "GUE":
            return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)
        # GSE
        return (2 ** 18 / (3 ** 6 * np.pi ** 3)) * s ** 4 * np.exp(-64 * s ** 2 / (9 * np.pi))

    def poisson_pdf(self, s: np.ndarray) -> np.ndarray:
        return np.exp(-np.asarray(s, dtype=float))

    def ks_test_goe(self, eigenvalues: np.ndarray) -> Tuple[float, float]:
        """KS test against GOE Wigner surmise. Returns (statistic, p-value)."""
        s = self.compute_spacings(eigenvalues)
        # GOE CDF: 1 - exp(-π s²/4)
        stat, pval = kstest(s, lambda x: 1.0 - np.exp(-np.pi * x ** 2 / 4))
        return float(stat), float(pval)

    def ks_test_poisson(self, eigenvalues: np.ndarray) -> Tuple[float, float]:
        """KS test against Poisson level spacing P(s) = exp(-s)."""
        s = self.compute_spacings(eigenvalues)
        stat, pval = kstest(s, "expon")
        return float(stat), float(pval)

    def classify(self, eigenvalues: np.ndarray, alpha: float = 0.05) -> str:
        """Classify eigenvalue spectrum as 'GOE', 'Poisson', or 'Intermediate'."""
        _, p_goe    = self.ks_test_goe(eigenvalues)
        _, p_poisson = self.ks_test_poisson(eigenvalues)
        if p_goe > alpha and p_poisson < alpha:
            return "GOE"
        if p_poisson > alpha and p_goe < alpha:
            return "Poisson"
        return "Intermediate"
