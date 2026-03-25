from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.polynomial.hermite import hermgauss
class FisherAnalyticCalculator:
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
        if q_star is None:
            q_star = self.fixed_point_variance()
        x, w = hermgauss(self.n_quadrature)
        z    = np.sqrt(2.0 * q_star) * x
        dphi = self._dphi(z)
        return float(self.sigma_w2 * np.dot(w, dphi ** 2) / np.sqrt(np.pi))
    def analytic_correlation_length(self) -> float:
        c = self.chi1()
        if c >= 1.0:
            return float("inf")
        return float(-1.0 / np.log(c))
    def analytic_fisher_trace(self, n_features: int) -> float:
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