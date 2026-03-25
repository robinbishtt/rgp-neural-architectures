from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit
@dataclass
class ScalingFitResult:
    law:         str              
    coefficients: np.ndarray     
    r2:          float
    rmse:        float
    coeff_ci:    np.ndarray      
    n_points:    int
def _log_law(x, A, B):
    return A * np.log(np.maximum(x, 1e-12)) + B
def _power_law(x, A, alpha, B):
    return A * np.power(np.maximum(x, 1e-12), alpha) + B
def _linear_law(x, A, B):
    return A * x + B
class ScalingLawFitter:
    def fit_logarithmic(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> ScalingFitResult:
        x, y = np.asarray(x, float), np.asarray(y, float)
        popt, pcov = curve_fit(_log_law, x, y, p0=[1.0, 0.0], maxfev=10_000)
        return self._build_result("logarithmic", popt, pcov, x, y,
                                  lambda x_: _log_law(x_, *popt))
    def fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> ScalingFitResult:
        x, y  = np.asarray(x, float), np.asarray(y, float)
        popt, pcov = curve_fit(_power_law, x, y,
                               p0=[1.0, 1.0, 0.0], maxfev=10_000,
                               bounds=([-np.inf, 0.0, -np.inf],
                                       [np.inf, 10.0, np.inf]))
        return self._build_result("power", popt, pcov, x, y,
                                  lambda x_: _power_law(x_, *popt))
    def fit_linear(
        self, x: np.ndarray, y: np.ndarray
    ) -> ScalingFitResult:
        x, y  = np.asarray(x, float), np.asarray(y, float)
        popt, pcov = curve_fit(_linear_law, x, y, p0=[1.0, 0.0])
        return self._build_result("linear", popt, pcov, x, y,
                                  lambda x_: _linear_law(x_, *popt))
    def _build_result(
        self,
        law: str,
        popt: np.ndarray,
        pcov: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        predict: Callable,
    ) -> ScalingFitResult:
        y_hat  = predict(x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = float(1.0 - ss_res / (ss_tot + 1e-12))
        rmse   = float(np.sqrt(ss_res / len(y)))
        perr   = np.sqrt(np.diag(pcov)) * 1.96
        ci     = np.column_stack([popt - perr, popt + perr])
        return ScalingFitResult(
            law=law, coefficients=popt, r2=r2, rmse=rmse,
            coeff_ci=ci, n_points=len(x),
        )