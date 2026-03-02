"""
src/core/spectral/wigner_semicircle.py

Wigner Semicircle distribution for eigenvalue density of large symmetric
random matrices (Gaussian Orthogonal Ensemble / Gaussian Unitary Ensemble).

ρ_sc(λ) = (2 / πR²) √(R² - λ²),  |λ| ≤ R,  R = 2σ√N

Reference: Wigner, E.P. (1955). Characteristic vectors of bordered matrices
           with infinite dimensions. Ann. Math., 62:548–564.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import kstest
from scipy.integrate import cumulative_trapezoid


class WignerSemicircleDistribution:
    """
    Wigner semicircle law for the empirical spectral distribution of large
    real symmetric (GOE) or Hermitian (GUE) random matrices.

    The density is supported on [-R, R] with:
        ρ_sc(λ) = (2 / πR²) √(R² - λ²)
        R        = 2σ√N  (spectral radius)

    At criticality, the Jacobian spectrum of a neural network layer approaches
    this distribution in the large-width limit.
    """

    def __init__(self, R: float) -> None:
        """
        Args:
            R: spectral radius (bulk edge), R = 2σ√N for N×N GOE matrix.
        """
        if R <= 0:
            raise ValueError(f"Spectral radius R must be positive, got {R}")
        self.R = R

    @classmethod
    def from_variance(cls, sigma2: float, N: int) -> "WignerSemicircleDistribution":
        """
        Construct from matrix entry variance σ² and dimension N.

        Args:
            sigma2: variance of off-diagonal entries
            N:      matrix dimension

        Returns:
            WignerSemicircleDistribution with R = 2σ√N
        """
        R = 2.0 * np.sqrt(sigma2 * N)
        return cls(R=R)

    def pdf(self, lam: np.ndarray) -> np.ndarray:
        """
        Semicircle density ρ_sc(λ).

        Args:
            lam: eigenvalue(s)

        Returns:
            Density values, same shape as lam
        """
        lam  = np.asarray(lam, dtype=float)
        rho  = np.zeros_like(lam)
        mask = np.abs(lam) < self.R
        rho[mask] = (2.0 / (np.pi * self.R ** 2)) * np.sqrt(self.R ** 2 - lam[mask] ** 2)
        return rho

    def cdf(self, lam: np.ndarray) -> np.ndarray:
        """Exact CDF: F(λ) = 1/2 + (λ√(R²-λ²))/(πR²) + arcsin(λ/R)/π"""
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
        """
        KS test against empirical eigenvalue sample.

        Returns:
            (ks_statistic, p_value)
        """
        ev = np.asarray(empirical_eigenvalues, dtype=float)
        ev = ev[np.abs(ev) <= self.R * 1.1]
        if len(ev) < 5:
            return 1.0, 0.0
        stat, pval = kstest(ev, lambda x: self.cdf(x))
        return float(stat), float(pval)

    def sample_goe(self, N: int, rng=None) -> np.ndarray:
        """
        Sample eigenvalues from an N×N Gaussian Orthogonal Ensemble (GOE) matrix.

        For the GOE with entry variance σ² = 1/N, R = 2 and the semicircle
        law holds as N → ∞.
        """
        rng = rng or np.random.default_rng()
        sigma = self.R / (2.0 * np.sqrt(N))
        A     = rng.standard_normal((N, N)) * sigma
        M     = (A + A.T) / np.sqrt(2.0)
        return np.linalg.eigvalsh(M)
 