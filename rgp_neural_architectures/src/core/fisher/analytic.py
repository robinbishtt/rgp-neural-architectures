"""
src/core/fisher/analytic.py

Closed-form Fisher information calculations for specific architectures.

For networks at critical initialization with specific nonlinearities,
certain quantities (χ₁, q*, the fixed-point variance) have analytic forms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.polynomial.hermite import hermgauss


class FisherAnalyticCalculator:
    """
    Analytic Fisher information computations for shallow/single-layer networks
    and networks at the mean-field fixed point.

    For a single linear layer with Gaussian weights W ~ N(0, σ_w²/N):
        F_ij = (1/N) δ_ij * E[h²]

    At the mean-field fixed point (σ_w², σ_b²), the Fisher metric
    has a closed-form spectrum derivable from χ₁.
    """

    def __init__(
        self,
        sigma_w2:     float = 1.0,
        sigma_b2:     float = 0.05,
        nonlinearity: str   = "tanh",
        n_quadrature: int   = 50,
    ) -> None:
        self.sigma_w2     = sigma_w2
        self.sigma_b2     = sigma_b2
        self.nonlinearity = nonlinearity
        self.n_quadrature = n_quadrature

    def fixed_point_variance(self, tol: float = 1e-8, max_iter: int = 500) -> float:
        """
        Compute the mean-field fixed point q* via iteration:
          q* = σ_w² E[φ(√(2q*) t)²] + σ_b²   (Gauss-Hermite quadrature)
        """
        x, w = hermgauss(self.n_quadrature)
        q = 1.0
        for _ in range(max_iter):
            z = np.sqrt(2.0 * q) * x
            phi = self._phi(z)
            q_new = self.sigma_w2 * np.dot(w, phi ** 2) / np.sqrt(np.pi) + self.sigma_b2
            if abs(q_new - q) < tol:
                return float(q_new)
            q = q_new
        return float(q)

    def chi1(self, q_star: Optional[float] = None) -> float:
        """
        χ₁ = σ_w² E[φ'(z)²]   z ~ N(0, q*)
        """
        if q_star is None:
            q_star = self.fixed_point_variance()
        x, w = hermgauss(self.n_quadrature)
        z    = np.sqrt(2.0 * q_star) * x
        dphi = self._dphi(z)
        return float(self.sigma_w2 * np.dot(w, dphi ** 2) / np.sqrt(np.pi))

    def analytic_correlation_length(self) -> float:
        """
        ξ_analytic = −1 / log(χ₁)   for χ₁ < 1 (ordered phase)
        """
        c = self.chi1()
        if c >= 1.0:
            return float("inf")
        return float(-1.0 / np.log(c))

    def analytic_fisher_trace(self, n_features: int) -> float:
        """
        Analytic trace of Fisher matrix for a single linear layer.
        Tr(F) ≈ n_features * σ_w² * q*
        """
        q_star = self.fixed_point_variance()
        return float(n_features * self.sigma_w2 * q_star)

    def _phi(self, z: np.ndarray) -> np.ndarray:
        if self.nonlinearity == "tanh":
            return np.tanh(z)
        if self.nonlinearity == "relu":
            return np.maximum(z, 0.0)
        return np.tanh(z)

    def _dphi(self, z: np.ndarray) -> np.ndarray:
        if self.nonlinearity == "tanh":
            return 1.0 - np.tanh(z) ** 2
        if self.nonlinearity == "relu":
            return (z > 0).astype(float)
        return 1.0 - np.tanh(z) ** 2
