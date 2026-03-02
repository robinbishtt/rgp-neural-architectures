"""
src/core/correlation/exponential_decay_fitter.py

ExponentialDecayFitter: fits the exponential decay law
    ξ(k) = ξ₀ · exp(−k / k_c)
to measured per-layer correlation lengths, extracting the initial
correlation length ξ₀ and the characteristic decay depth k_c.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


@dataclass
class ExponentialDecayFitResult:
    xi_0:        float                   # fitted initial correlation length
    k_c:         float                   # fitted decay depth constant
    r2:          float                   # coefficient of determination
    xi_0_ci:     Tuple[float, float]     # 95% confidence interval for xi_0
    k_c_ci:      Tuple[float, float]     # 95% confidence interval for k_c
    chi1:        float                   # criticality parameter chi1 = exp(-1/k_c)
    residuals:   np.ndarray             # per-layer fitting residuals


def _exp_decay(k: np.ndarray, xi_0: float, k_c: float) -> np.ndarray:
    return xi_0 * np.exp(-k / k_c)


class ExponentialDecayFitter:
    """
    Fits the RGP correlation length decay law ξ(k) = ξ₀ · exp(-k / k_c)
    to a sequence of per-layer correlation length measurements.

    The decay law is central to Hypothesis H1 (scale correspondence):
    each network layer k corresponds to RG scale s(k) ∝ ξ(k), so an
    exponentially decaying correlation length directly maps network depth
    to a hierarchy of physical scales.

    The criticality parameter χ₁ = exp(-1/k_c):
        χ₁ < 1  → ordered phase (exponential information contraction)
        χ₁ = 1  → critical point (power-law scaling)
        χ₁ > 1  → chaotic phase (exponential information amplification)
    """

    def __init__(
        self,
        p0_xi0: float         = 10.0,
        p0_kc:  float         = 20.0,
        maxfev: int           = 10_000,
    ) -> None:
        """
        Args:
            p0_xi0: initial guess for ξ₀ (default 10.0)
            p0_kc:  initial guess for k_c (default 20.0)
            maxfev: maximum function evaluations for scipy curve_fit
        """
        self.p0     = [p0_xi0, p0_kc]
        self.maxfev = maxfev

    def fit(
        self,
        layers:      np.ndarray,
        xi_values:   np.ndarray,
        weights:     Optional[np.ndarray] = None,
    ) -> ExponentialDecayFitResult:
        """
        Fit ξ(k) = ξ₀ · exp(−k / k_c) to measured correlation lengths.

        Args:
            layers:    array of layer indices k (must be non-negative)
            xi_values: measured correlation length ξ(k) per layer
            weights:   optional inverse-variance weights for weighted least-squares

        Returns:
            ExponentialDecayFitResult with fitted parameters and diagnostics
        """
        layers    = np.asarray(layers, dtype=float)
        xi_values = np.asarray(xi_values, dtype=float)
        sigma     = None if weights is None else 1.0 / (np.sqrt(weights) + 1e-12)

        bounds = ([0.0, 0.01], [np.inf, np.inf])
        try:
            popt, pcov = curve_fit(
                _exp_decay, layers, xi_values,
                p0=self.p0, sigma=sigma,
                bounds=bounds, maxfev=self.maxfev,
            )
        except RuntimeError:
            # fallback: log-linear regression
            log_xi     = np.log(xi_values + 1e-12)
            coeffs     = np.polyfit(layers, log_xi, 1)
            k_c_fit    = -1.0 / (coeffs[0] + 1e-12)
            xi_0_fit   = np.exp(coeffs[1])
            popt       = np.array([xi_0_fit, k_c_fit])
            pcov       = np.eye(2) * 1e6

        xi_0, k_c  = float(popt[0]), float(popt[1])
        xi_hat     = _exp_decay(layers, xi_0, k_c)
        residuals  = xi_values - xi_hat

        # R² computation
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((xi_values - xi_values.mean()) ** 2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-12)

        # 95% confidence intervals
        perr  = np.sqrt(np.diag(pcov)) * 1.96
        ci_xi = (xi_0 - perr[0], xi_0 + perr[0])
        ci_kc = (k_c - perr[1], k_c + perr[1])
        chi1  = float(np.exp(-1.0 / k_c))

        return ExponentialDecayFitResult(
            xi_0=xi_0, k_c=k_c, r2=float(r2),
            xi_0_ci=ci_xi, k_c_ci=ci_kc,
            chi1=chi1, residuals=residuals,
        )
 