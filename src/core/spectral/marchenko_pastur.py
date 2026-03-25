from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import kstest
from scipy.integrate import cumulative_trapezoid
class MarchenkoPasturDistribution:
    def __init__(self, beta: float, sigma2: float = 1.0) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta       = beta
        self.sigma2     = sigma2
        self.lam_minus  = sigma2 * (1.0 - np.sqrt(beta)) ** 2
        self.lam_plus   = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    @property
    def support(self) -> Tuple[float, float]:
        return self.lam_minus, self.lam_plus
    @property
    def mean(self) -> float:
        return float(self.sigma2)
    @property
    def variance(self) -> float:
        return float(self.sigma2 ** 2 * self.beta)
    def pdf(self, lam: np.ndarray) -> np.ndarray:
        lam  = np.asarray(lam, dtype=float)
        rho  = np.zeros_like(lam)
        mask = (lam > self.lam_minus) & (lam < self.lam_plus)
        l    = lam[mask]
        rho[mask] = (
            np.sqrt((self.lam_plus - l) * (l - self.lam_minus))
            / (2.0 * np.pi * self.sigma2 * self.beta * l)
        )
        return rho
    def cdf(self, lam: np.ndarray, n_points: int = 5000) -> np.ndarray:
        lam  = np.asarray(lam, dtype=float)
        x    = np.linspace(self.lam_minus * 0.99, self.lam_plus * 1.01, n_points)
        y    = self.pdf(x)
        cdf  = np.concatenate([[0.0], cumulative_trapezoid(y, x)])
        norm = max(cdf[-1], 1e-12)
        return np.interp(lam, x, cdf / norm)
    def ks_test(self, empirical_eigenvalues: np.ndarray) -> Tuple[float, float]:
        ev = np.asarray(empirical_eigenvalues, dtype=float)
        ev = ev[(ev >= self.lam_minus * 0.9) & (ev <= self.lam_plus * 1.1)]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)
    def sample_wishart(self, n: int, m: int, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        X   = rng.standard_normal((n, m)) * np.sqrt(self.sigma2)
        W   = X @ X.T / m
        return np.linalg.eigvalsh(W)