from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import kstest
class WignerSemicircleDistribution:
    def __init__(self, R: float = None, *, radius: float = None) -> None:
        r_val = R if R is not None else radius
        if r_val is None:
            raise ValueError("Provide R (positional) or radius= (keyword)")
        if r_val <= 0:
            raise ValueError(f"Spectral radius R must be positive, got {r_val}")
        self.R = r_val
        self.radius = r_val  
    @classmethod
    def from_variance(cls, sigma2: float, N: int) -> "WignerSemicircleDistribution":
        R = 2.0 * np.sqrt(sigma2 * N)
        return cls(R=R)
    def pdf(self, lam: np.ndarray) -> np.ndarray:
        lam  = np.asarray(lam, dtype=float)
        rho  = np.zeros_like(lam)
        mask = np.abs(lam) < self.R
        rho[mask] = (2.0 / (np.pi * self.R ** 2)) * np.sqrt(self.R ** 2 - lam[mask] ** 2)
        return rho
    def cdf(self, lam: np.ndarray) -> np.ndarray:
        lam  = np.asarray(lam, dtype=float)
        out  = np.zeros_like(lam)
        mask = (lam > -self.R) & (lam < self.R)
        l    = lam[mask]
        out[mask] = (
            0.5
            + (l * np.sqrt(self.R ** 2 - l ** 2)) / (np.pi * self.R ** 2)
            + np.arcsin(l / self.R) / np.pi
        )
        out[lam >= self.R] = 1.0
        return out
    def ks_test(self, empirical_eigenvalues: np.ndarray) -> Tuple[float, float]:
        ev = np.asarray(empirical_eigenvalues, dtype=float)
        ev = ev[np.abs(ev) <= self.R * 1.1]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)
    def sample_goe(self, N: int, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        sigma = self.R / (2.0 * np.sqrt(N))
        A     = rng.standard_normal((N, N)) * sigma
        M     = (A + A.T) / np.sqrt(2.0)
        return np.linalg.eigvalsh(M)