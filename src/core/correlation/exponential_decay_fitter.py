from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
@dataclass
class ExponentialDecayFitResult:
    xi_0:        float                   
    k_c:         float                   
    r2:          float                   
    xi_0_ci:     Tuple[float, float]     
    k_c_ci:      Tuple[float, float]     
    chi1:        float                   
    residuals:   np.ndarray             
def _exp_decay(k: np.ndarray, xi_0: float, k_c: float) -> np.ndarray:
    return xi_0 * np.exp(-k / k_c)
class ExponentialDecayFitter:
    def __init__(
        self,
        p0_xi0: float         = 10.0,
        p0_kc:  float         = 20.0,
        maxfev: int           = 10_000,
    ) -> None:
        self.p0     = [p0_xi0, p0_kc]
        self.maxfev = maxfev
    def fit(
        self,
        layers_or_xi: np.ndarray,
        xi_values:    np.ndarray = None,
        weights:      Optional[np.ndarray] = None,
    ) -> ExponentialDecayFitResult:
        if xi_values is None:
            xi_values = np.asarray(layers_or_xi, dtype=float)
            layers    = np.arange(len(xi_values), dtype=float)
        else:
            layers    = np.asarray(layers_or_xi, dtype=float)
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
            log_xi     = np.log(xi_values + 1e-12)
            coeffs     = np.polyfit(layers, log_xi, 1)
            k_c_fit    = -1.0 / (coeffs[0] + 1e-12)
            xi_0_fit   = np.exp(coeffs[1])
            popt       = np.array([xi_0_fit, k_c_fit])
            pcov       = np.eye(2) * 1e6
        xi_0, k_c  = float(popt[0]), float(popt[1])
        xi_hat     = _exp_decay(layers, xi_0, k_c)
        residuals  = xi_values - xi_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((xi_values - xi_values.mean()) ** 2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-12)
        perr  = np.sqrt(np.diag(pcov)) * 1.96
        ci_xi = (xi_0 - perr[0], xi_0 + perr[0])
        ci_kc = (k_c - perr[1], k_c + perr[1])
        chi1  = float(np.exp(-1.0 / k_c))
        return ExponentialDecayFitResult(
            xi_0=xi_0, k_c=k_c, r2=float(r2),
            xi_0_ci=ci_xi, k_c_ci=ci_kc,
            chi1=chi1, residuals=residuals,
        )