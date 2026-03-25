from __future__ import annotations
from typing import Tuple
import numpy as np
class TracyWidomDistribution:
    def __init__(self, beta: int = 1) -> None:
        if beta not in (1, 2):
            raise ValueError(f"beta must be 1 (GOE) or 2 (GUE), got {beta}")
        self.beta = beta
        self._precompute_table()
    def _precompute_table(self, n_points: int = 2000) -> None:
        self._s_grid   = np.linspace(-8.0, 4.0, n_points)
        if self.beta == 1:
            mu, sigma = -1.2065, 1.2780
        else:
            mu, sigma = -1.7711, 0.9018
        from scipy.stats import norm
        self._cdf_table = norm.cdf((self._s_grid - mu) / sigma)
    def cdf(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.interp(s, self._s_grid, self._cdf_table)
    def pdf(self, s: np.ndarray) -> np.ndarray:
        s    = np.asarray(s, dtype=float)
        dcdf = np.gradient(self._cdf_table, self._s_grid)
        return np.interp(s, self._s_grid, dcdf)
    def scaled_statistic(
        self, lambda_max: float, R: float, N: int
    ) -> float:
        return float(N ** (2.0 / 3.0) * (lambda_max / R - 1.0))
    def ks_test_largest_eigenvalues(
        self, largest_eigenvalues: np.ndarray, R: float, N: int
    ) -> Tuple[float, float]:
        from scipy.stats import kstest
        scaled = np.array([
            self.scaled_statistic(lm, R, N)
            for lm in largest_eigenvalues
        ])
        stat, pval = kstest(scaled, self.cdf)
        return float(stat), float(pval)