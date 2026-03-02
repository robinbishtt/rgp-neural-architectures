"""
src/core/spectral/marchenko_pastur.py

Marchenko-Pastur distribution for eigenvalue density of large Wishart matrices.

ρ_MP(λ) = √[(λ₊ - λ)(λ - λ₋)] / (2π σ² β λ)
λ± = σ²(1 ± √β)²,  β = N/M (aspect ratio)

Reference: Marchenko & Pastur (1967). Distribution of eigenvalues for some
           sets of random matrices. Math. USSR-Sb., 1(4):457–483.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import kstest
from scipy.integrate import cumulative_trapezoid


class MarchenkoPasturDistribution:
    """
    Theoretical eigenvalue density for large Wishart matrices W = (1/M) X Xᵀ
    where X is N×M with i.i.d. N(0, σ²) entries and β = N/M.

    The bulk spectrum is supported on [λ₋, λ₊] with
        λ± = σ²(1 ± √β)²

    When β > 1, there is additionally a point mass at 0 of weight (1 - 1/β).
    """

    def __init__(self, beta: float, sigma2: float = 1.0) -> None:
        """
        Args:
            beta:   aspect ratio N/M (must be positive)
            sigma2: variance of matrix entries (default 1.0)
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta       = beta
        self.sigma2     = sigma2
        self.lam_minus  = sigma2 * (1.0 - np.sqrt(beta)) ** 2
        self.lam_plus   = sigma2 * (1.0 + np.sqrt(beta)) ** 2

    @property
    def support(self) -> Tuple[float, float]:
        """Bulk support [λ₋, λ₊]."""
        return self.lam_minus, self.lam_plus

    @property
    def mean(self) -> float:
        """Population mean: σ²."""
        return float(self.sigma2)

    @property
    def variance(self) -> float:
        """Population variance: σ⁴ β."""
        return float(self.sigma2 ** 2 * self.beta)

    def pdf(self, lam: np.ndarray) -> np.ndarray:
        """
        Probability density function evaluated at eigenvalue(s) lam.

        Args:
            lam: eigenvalue(s), scalar or array

        Returns:
            ρ(λ), same shape as lam
        """
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
        """Cumulative distribution function via numerical integration."""
        lam  = np.asarray(lam, dtype=float)
        x    = np.linspace(self.lam_minus * 0.99, self.lam_plus * 1.01, n_points)
        y    = self.pdf(x)
        cdf  = np.concatenate([[0.0], cumulative_trapezoid(y, x)])
        norm = max(cdf[-1], 1e-12)
        return np.interp(lam, x, cdf / norm)

    def ks_test(self, empirical_eigenvalues: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test: do empirical eigenvalues follow MP?

        Args:
            empirical_eigenvalues: 1-D array of measured eigenvalues

        Returns:
            (ks_statistic, p_value). p_value < 0.05 → significant deviation.
        """
        ev = np.asarray(empirical_eigenvalues, dtype=float)
        ev = ev[(ev >= self.lam_minus * 0.9) & (ev <= self.lam_plus * 1.1)]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)

    def sample_wishart(self, n: int, m: int, rng=None) -> np.ndarray:
        """
        Sample eigenvalues from a random Wishart matrix of shape (n, n)
        formed from an n×m Gaussian matrix X with σ²-scaled entries.

        Args:
            n:   number of rows
            m:   number of columns
            rng: numpy RandomGenerator (optional)

        Returns:
            eigenvalues: sorted array of n eigenvalues
        """
        rng = rng or np.random.default_rng()
        X   = rng.standard_normal((n, m)) * np.sqrt(self.sigma2)
        W   = X @ X.T / m
        return np.linalg.eigvalsh(W)
 