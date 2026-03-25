from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss
def _fixed_point_variance(
    sigma_w2: float,
    sigma_b2: float,
    nonlinearity: str = "tanh",
    n_points: int = 50,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> float:
    x_gh, w_gh = hermgauss(n_points)
    q = 1.0  
    for _ in range(max_iter):
        z     = np.sqrt(2.0 * max(q, 1e-12)) * x_gh
        phi_z = _phi(z, nonlinearity)
        q_new = sigma_w2 * np.dot(w_gh, phi_z ** 2) / np.sqrt(np.pi) + sigma_b2
        if abs(q_new - q) < tol:
            return float(q_new)
        q = q_new
    return float(q)
def _phi(z: np.ndarray, nonlinearity: str) -> np.ndarray:
    if nonlinearity == "tanh":
        return np.tanh(z)
    if nonlinearity == "relu":
        return np.maximum(z, 0.0)
    if nonlinearity == "gelu":
        from scipy.special import erf
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    raise ValueError(f"Unknown nonlinearity: {nonlinearity!r}")
def _dphi(z: np.ndarray, nonlinearity: str) -> np.ndarray:
    if nonlinearity == "tanh":
        return 1.0 - np.tanh(z) ** 2
    if nonlinearity == "relu":
        return (z > 0).astype(float)
    if nonlinearity == "gelu":
        from scipy.special import erf
        cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        return cdf + z * pdf
    raise ValueError(f"Unknown nonlinearity: {nonlinearity!r}")
def chi1_gauss_hermite(
    sigma_w2:     float,
    nonlinearity: str   = "tanh",
    n_points:     int   = 50,
    q_star:       float = None,
    sigma_b2:     float = 0.09,   
) -> float:
    if q_star is None:
        q_star = _fixed_point_variance(sigma_w2, sigma_b2, nonlinearity, n_points)
    x_gh, w_gh = hermgauss(n_points)
    z    = np.sqrt(2.0 * max(q_star, 1e-12)) * x_gh
    dphi = _dphi(z, nonlinearity)
    return float(sigma_w2 * np.dot(w_gh, dphi ** 2) / np.sqrt(np.pi))
def critical_sigma_w2(
    nonlinearity: str   = "tanh",
    n_points:     int   = 50,
    sigma_b2:     float = 0.09,
    tol:          float = 1e-8,
) -> float:
    lo, hi = 0.01, 20.0
    for _ in range(100):
        mid  = 0.5 * (lo + hi)
        chi1 = chi1_gauss_hermite(mid, nonlinearity, n_points, sigma_b2=sigma_b2)
        if abs(chi1 - 1.0) < tol:
            return float(mid)
        if chi1 < 1.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
def ordered_sigma_w_for_target_chi1(
    target_chi1:  float,
    nonlinearity: str   = "tanh",
    sigma_b2:     float = 0.09,
    n_points:     int   = 50,
    tol:          float = 1e-8,
) -> float:
    assert 0.0 < target_chi1 < 1.0, "target_chi1 must be in (0, 1) for ordered phase"
    sw2_crit = critical_sigma_w2(nonlinearity, n_points, sigma_b2)
    lo, hi = 0.01, sw2_crit
    for _ in range(100):
        mid  = 0.5 * (lo + hi)
        chi1 = chi1_gauss_hermite(mid, nonlinearity, n_points, sigma_b2=sigma_b2)
        if abs(chi1 - target_chi1) < tol:
            return float(mid ** 0.5)
        if chi1 > target_chi1:
            hi = mid
        else:
            lo = mid
    return float((0.5 * (lo + hi)) ** 0.5)
def recommended_sigma_w(k_c_target: float, sigma_b: float = 0.3) -> float:
    if k_c_target <= 0:
        raise ValueError(f"k_c_target must be positive, got {k_c_target}")
    target_chi1 = float(np.exp(-1.0 / k_c_target))
    return ordered_sigma_w_for_target_chi1(
        target_chi1, sigma_b2=sigma_b ** 2
    )