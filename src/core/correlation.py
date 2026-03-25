from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss
def chi1_gauss_hermite(
    sigma_w2: float,
    nonlinearity: str = "tanh",
    n_points: int = 50,
    q_star: float = 1.0,
) -> float:
    x_gh, w_gh = hermgauss(n_points)
    z = np.sqrt(2.0 * q_star) * x_gh
    if nonlinearity == "tanh":
        phi_prime = 1.0 - np.tanh(z) ** 2
    elif nonlinearity == "relu":
        phi_prime = (z > 0).astype(float)
    elif nonlinearity == "gelu":
        from scipy.special import erf
        phi_prime = 0.5 * (1.0 + erf(z / np.sqrt(2.0))) +                    z * np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity!r}")
    integrand = phi_prime ** 2
    integral  = np.dot(w_gh, integrand) / np.sqrt(np.pi)
    return float(sigma_w2 * integral)
def critical_sigma_w2(
    nonlinearity: str = "tanh",
    n_points: int = 50,
    tol: float = 1e-6,
) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c   = chi1_gauss_hermite(mid, nonlinearity, n_points)
        if abs(c - 1.0) < tol:
            return mid
        if c < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
class TwoPointCorrelation:
    def __init__(
        self,
        sigma_w2: float = 1.0,
        sigma_b2: float = 0.05,
        nonlinearity: str = "tanh",
        n_quadrature: int = 50,
    ) -> None:
        self.sigma_w2    = sigma_w2
        self.sigma_b2    = sigma_b2
        self.nonlinearity = nonlinearity
        self.n_quadrature = n_quadrature
    def propagate(
        self,
        q11: float,  
        q12: float,  
        q22: float,  
    ):
        x_gh, w_gh = hermgauss(self.n_quadrature)
        def _integrate(fn):
            return float(np.dot(w_gh, fn(x_gh)) / np.sqrt(np.pi))
        sigma = np.sqrt(max(2.0 * q11, 1e-12))
        if self.nonlinearity == "tanh":
            def fn11(t):
                return np.tanh(sigma * t) ** 2
        elif self.nonlinearity == "relu":
            def fn11(t):
                return np.maximum(sigma * t, 0) ** 2
        else:
            def fn11(t):
                return np.tanh(sigma * t) ** 2
        new_q11 = self.sigma_w2 * _integrate(fn11) + self.sigma_b2
        new_q12 = new_q11 * (q12 / max(q11, 1e-12))  
        new_q22 = new_q11
        return new_q11, new_q12, new_q22
    def run(self, n_layers: int, c12_init: float = 0.9) -> np.ndarray:
        q11, q12, q22 = 1.0, c12_init, 1.0
        c12_vals = [c12_init]
        for _ in range(n_layers):
            q11, q12, q22 = self.propagate(q11, q12, q22)
            c12 = q12 / np.sqrt(max(q11 * q22, 1e-12))
            c12_vals.append(float(np.clip(c12, -1.0, 1.0)))
        return np.array(c12_vals)