from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.stats import gaussian_kde, kstest
class MarchenkoPasturDistribution:
    def __init__(self, beta: float, sigma2: float = 1.0) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta   = beta
        self.sigma2 = sigma2
        self.lam_minus = sigma2 * (1.0 - np.sqrt(beta)) ** 2
        self.lam_plus  = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    def pdf(self, lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        rho = np.zeros_like(lam)
        mask = (lam > self.lam_minus) & (lam < self.lam_plus)
        l = lam[mask]
        rho[mask] = (
            np.sqrt((self.lam_plus - l) * (l - self.lam_minus))
            / (2.0 * np.pi * self.sigma2 * self.beta * l)
        )
        return rho
    def cdf(self, lam: np.ndarray, n_points: int = 5000) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        x   = np.linspace(self.lam_minus * 0.99, self.lam_plus * 1.01, n_points)
        y   = self.pdf(x)
        cdf_vals = np.concatenate([[0.0], np.cumsum(y[:-1] * np.diff(x))])
        cdf_norm = cdf_vals / max(cdf_vals[-1], 1e-12)
        return np.interp(lam, x, cdf_norm)
    def ks_test(self, empirical_eigenvalues: np.ndarray) -> Tuple[float, float]:
        ev = np.asarray(empirical_eigenvalues)
        ev = ev[(ev >= self.lam_minus * 0.9) & (ev <= self.lam_plus * 1.1)]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)
    def sample_wishart(self, n: int, m: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((n, m)) * np.sqrt(self.sigma2)
        W = X @ X.T / m
        return np.linalg.eigvalsh(W)
class WignerSemicircleDistribution:
    def __init__(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = radius
    def pdf(self, lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        R   = self.radius
        rho = np.zeros_like(lam)
        mask = np.abs(lam) < R
        rho[mask] = (2.0 / (np.pi * R ** 2)) * np.sqrt(R ** 2 - lam[mask] ** 2)
        return rho
    def ks_test(self, empirical_eigenvalues: np.ndarray) -> Tuple[float, float]:
        ev = np.asarray(empirical_eigenvalues)
        R  = self.radius
        ev = ev[np.abs(ev) < R * 1.1]
        if len(ev) < 5:
            return 1.0, 0.0
        def cdf(x):
            x = np.clip(x, -R, R)
            return 0.5 + (x * np.sqrt(R**2 - x**2) + R**2 * np.arcsin(x / R)) / (np.pi * R**2)
        stat, pval = kstest(ev, cdf)
        return float(stat), float(pval)
class TracyWidomDistribution:
    def __init__(self) -> None:
        self._s_grid = np.linspace(-6.0, 4.0, 1000)
        self._cdf_grid = 1.0 / (1.0 + np.exp(-1.2 * (self._s_grid + 1.5)))
    def cdf(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s, dtype=float), self._s_grid, self._cdf_grid)
    def pdf(self, s: np.ndarray) -> np.ndarray:
        s    = np.asarray(s, dtype=float)
        cdf  = self.cdf(s)
        ds   = s[1] - s[0] if s.ndim > 0 and len(s) > 1 else 0.01
        return np.gradient(cdf, ds) if s.ndim > 0 and len(s) > 1 else np.zeros_like(s)
def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bw_method: float = 0.25,
    n_points: int = 500,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ev = np.asarray(eigenvalues, dtype=float)
    if xlim is None:
        xlim = (ev.min() * 0.9, ev.max() * 1.1)
    x_grid = np.linspace(xlim[0], xlim[1], n_points)
    kde    = gaussian_kde(ev, bw_method=bw_method)
    return x_grid, kde(x_grid)