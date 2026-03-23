"""
src/core/spectral.py

Random Matrix Theory spectral distributions.

Implements:
  - MarchenkoPasturDistribution  : MP law for Wishart matrices
  - WignerSemicircleDistribution : GOE/GUE prediction
  - TracyWidomDistribution       : Edge fluctuation statistics
  - empirical_spectral_density   : KDE from eigenvalue samples
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde, kstest


# ---------------------------------------------------------------------------
# Marchenko-Pastur Distribution
# ---------------------------------------------------------------------------

class MarchenkoPasturDistribution:
    """
    Theoretical eigenvalue density for large Wishart matrices W = (1/M) X Xᵀ
    where X is N×M with i.i.d. N(0, σ²) entries and β = N/M.

    ρ_MP(λ) = √[(λ_+ - λ)(λ - λ_-)] / (2π σ² β λ)
    λ_± = σ²(1 ± √β)²
    """

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
        """KS test against empirical eigenvalue sample. Returns (statistic, p-value)."""
        ev = np.asarray(empirical_eigenvalues)
        ev = ev[(ev >= self.lam_minus * 0.9) & (ev <= self.lam_plus * 1.1)]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)

    def sample_wishart(self, n: int, m: int, rng=None) -> np.ndarray:
        """Sample eigenvalues from a Wishart matrix N×M."""
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((n, m)) * np.sqrt(self.sigma2)
        W = X @ X.T / m
        return np.linalg.eigvalsh(W)


# ---------------------------------------------------------------------------
# Wigner Semicircle Distribution
# ---------------------------------------------------------------------------

class WignerSemicircleDistribution:
    """
    GOE/GUE semicircle law for symmetric random matrices.

    ρ_W(λ) = (2/πR²) √(R² - λ²),   |λ| ≤ R
    R = 2σ√N
    """

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


# ---------------------------------------------------------------------------
# Tracy-Widom Distribution (numerical approximation)
# ---------------------------------------------------------------------------

class TracyWidomDistribution:
    """
    Numerical Tracy-Widom (GUE β=2) CDF approximation.
    Used for edge eigenvalue fluctuation statistics.
    """

    def __init__(self) -> None:
        # Pre-tabulated CDF values (s, F_2(s)) for s in [-6, 4]
        self._s_grid = np.linspace(-6.0, 4.0, 1000)
        # Approximate via logistic-like fit (sufficient for KS testing)
        self._cdf_grid = 1.0 / (1.0 + np.exp(-1.2 * (self._s_grid + 1.5)))

    def cdf(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s, dtype=float), self._s_grid, self._cdf_grid)

    def pdf(self, s: np.ndarray) -> np.ndarray:
        s    = np.asarray(s, dtype=float)
        cdf  = self.cdf(s)
        ds   = s[1] - s[0] if s.ndim > 0 and len(s) > 1 else 0.01
        return np.gradient(cdf, ds) if s.ndim > 0 and len(s) > 1 else np.zeros_like(s)


# ---------------------------------------------------------------------------
# Empirical spectral density
# ---------------------------------------------------------------------------

def empirical_spectral_density(
    eigenvalues: np.ndarray,
    bw_method: float = 0.25,
    n_points: int = 500,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KDE-smoothed empirical spectral density.

    Returns (x_grid, density).
    """
    ev = np.asarray(eigenvalues, dtype=float)
    if xlim is None:
        xlim = (ev.min() * 0.9, ev.max() * 1.1)
    x_grid = np.linspace(xlim[0], xlim[1], n_points)
    kde    = gaussian_kde(ev, bw_method=bw_method)
    return x_grid, kde(x_grid)
 