from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
@dataclass
class WidthScalingResult:
    widths:           np.ndarray     
    observables:      np.ndarray     
    critical_width:   Optional[float]
    scaling_exponent: Optional[float]  
    correlation_nu:   Optional[float]  
    r2:               float
class WidthScalingAnalyzer:
    def fit_power_law(
        self,
        widths:      np.ndarray,
        observables: np.ndarray,
    ) -> WidthScalingResult:
        widths      = np.asarray(widths, float)
        observables = np.asarray(observables, float)
        def model(N, A, gamma, O_inf):
            return A * N ** (-gamma) + O_inf
        from scipy.optimize import curve_fit
        try:
            popt, pcov = curve_fit(
                model, widths, observables,
                p0=[1.0, 0.5, observables[-1]],
                bounds=([0, 0, -np.inf], [np.inf, 5.0, np.inf]),
                maxfev=10_000,
            )
            A, gamma, O_inf = popt
            y_hat = model(widths, *popt)
            ss_res = np.sum((observables - y_hat) ** 2)
            ss_tot = np.sum((observables - observables.mean()) ** 2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        except Exception:
            gamma = 0.0
            r2    = 0.0
        return WidthScalingResult(
            widths=widths,
            observables=observables,
            critical_width=None,
            scaling_exponent=float(gamma),
            correlation_nu=None,
            r2=r2,
        )
    def infinite_width_extrapolation(
        self,
        widths:      np.ndarray,
        observables: np.ndarray,
    ) -> float:
        widths      = np.asarray(widths, float)
        observables = np.asarray(observables, float)
        inv_N = 1.0 / widths
        coeffs = np.polyfit(inv_N, observables, 1)
        return float(coeffs[1])  