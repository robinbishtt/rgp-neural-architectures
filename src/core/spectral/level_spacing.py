from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
from scipy.stats import kstest
class LevelSpacingDistribution:
    def compute_spacings(self, eigenvalues: np.ndarray) -> np.ndarray:
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
        return (2 ** 18 / (3 ** 6 * np.pi ** 3)) * s ** 4 * np.exp(-64 * s ** 2 / (9 * np.pi))
    def poisson_pdf(self, s: np.ndarray) -> np.ndarray:
        return np.exp(-np.asarray(s, dtype=float))
    def ks_test_goe(self, eigenvalues: np.ndarray) -> Tuple[float, float]:
        s = self.compute_spacings(eigenvalues)
        stat, pval = kstest(s, lambda x: 1.0 - np.exp(-np.pi * x ** 2 / 4))
        return float(stat), float(pval)
    def ks_test_poisson(self, eigenvalues: np.ndarray) -> Tuple[float, float]:
        s = self.compute_spacings(eigenvalues)
        stat, pval = kstest(s, "expon")
        return float(stat), float(pval)
    def classify(self, eigenvalues: np.ndarray, alpha: float = 0.05) -> str:
        _, p_goe    = self.ks_test_goe(eigenvalues)
        _, p_poisson = self.ks_test_poisson(eigenvalues)
        if p_goe > alpha and p_poisson < alpha:
            return "GOE"
        if p_poisson > alpha and p_goe < alpha:
            return "Poisson"
        return "Intermediate"