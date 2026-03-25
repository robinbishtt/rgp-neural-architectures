from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
@dataclass
class ScalingFitResult:
    slope:     float
    intercept: float
    r2:        float
    aic:       float
    model_name: str
    slope_ci:  Tuple[float, float] = (0.0, 0.0)
def _compute_aic(y_obs: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n    = len(y_obs)
    rss  = np.sum((y_obs - y_pred) ** 2)
    s2   = max(rss / n, 1e-12)
    log_lik = -0.5 * n * (np.log(2 * np.pi * s2) + 1.0)
    return 2.0 * n_params - 2.0 * log_lik
class DepthScalingFitter:
    def fit(
        self,
        xi_values: np.ndarray,
        l_min_values: np.ndarray,
    ) -> ScalingFitResult:
        log_xi = np.log(np.asarray(xi_values, dtype=float))
        l_min  = np.asarray(l_min_values, dtype=float)
        A     = np.vstack([log_xi, np.ones_like(log_xi)]).T
        coef, _, _, _ = np.linalg.lstsq(A, l_min, rcond=None)
        slope, intercept = coef
        l_pred = slope * log_xi + intercept
        ss_res = ((l_min - l_pred) ** 2).sum()
        ss_tot = ((l_min - l_min.mean()) ** 2).sum()
        r2     = 1.0 - ss_res / max(ss_tot, 1e-12)
        aic    = _compute_aic(l_min, l_pred, 2)
        return ScalingFitResult(
            slope=float(slope), intercept=float(intercept),
            r2=float(r2), aic=float(aic), model_name="logarithmic",
        )
class AICModelSelector:
    def select(
        self,
        xi_values: np.ndarray,
        l_min_values: np.ndarray,
    ) -> Dict[str, ScalingFitResult]:
        xi = np.asarray(xi_values, dtype=float)
        y  = np.asarray(l_min_values, dtype=float)
        results: Dict[str, ScalingFitResult] = {}
        log_xi = np.log(xi)
        A = np.vstack([log_xi, np.ones_like(log_xi)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = coef[0] * log_xi + coef[1]
        results["logarithmic"] = ScalingFitResult(
            slope=float(coef[0]), intercept=float(coef[1]),
            r2=float(1 - ((y - y_pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
            aic=float(_compute_aic(y, y_pred, 2)),
            model_name="logarithmic",
        )
        A = np.vstack([xi, np.ones_like(xi)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = coef[0] * xi + coef[1]
        results["linear"] = ScalingFitResult(
            slope=float(coef[0]), intercept=float(coef[1]),
            r2=float(1 - ((y - y_pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
            aic=float(_compute_aic(y, y_pred, 2)),
            model_name="linear",
        )
        try:
            def _pow(x, a, alpha, b): return a * x ** alpha + b
            popt, _ = curve_fit(_pow, xi, y, p0=[1.0, 0.5, 0.0], maxfev=5000)
            y_pred = _pow(xi, *popt)
            results["power_law"] = ScalingFitResult(
                slope=float(popt[1]), intercept=float(popt[2]),
                r2=float(1 - ((y - y_pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
                aic=float(_compute_aic(y, y_pred, 3)),
                model_name="power_law",
            )
        except Exception:
            pass
        return results
class DataCollapser:
    def collapse(
        self,
        xi_values: np.ndarray,
        widths: np.ndarray,
        acc_matrix: np.ndarray,
        xi_c: float,
        nu: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_all, y_all = [], []
        for i, xi in enumerate(xi_values):
            for j, N in enumerate(widths):
                x = (xi - xi_c) * N ** (1.0 / nu)
                y = acc_matrix[i, j]
                x_all.append(x)
                y_all.append(y)
        return np.array(x_all), np.array(y_all)
class CriticalExponentFitter:
    def fit(
        self,
        xi_values: np.ndarray,
        widths: np.ndarray,
        acc_matrix: np.ndarray,
        n_bootstrap: int = 200,
        rng=None,
    ) -> Tuple[float, float, float]:
        if rng is None:
            rng = np.random.default_rng(42)
        def _collapse_residual(nu_try: float) -> float:
            total = 0.0
            count = 0
            xi_c = xi_values[len(xi_values) // 2]
            for i, xi in enumerate(xi_values):
                for j, N in enumerate(widths):
                    x = (xi - xi_c) * N ** (1.0 / max(nu_try, 1e-3))
                    y_pred = 1.0 / (1.0 + np.exp(-x))
                    total += (acc_matrix[i, j] - y_pred) ** 2
                    count += 1
            return total / max(count, 1)
        res = minimize_scalar(_collapse_residual, bounds=(0.3, 4.0), method="bounded")
        nu_best = float(res.x)
        boots = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(xi_values), len(xi_values))
            xi_b   = xi_values[idx]
            acc_b  = acc_matrix[idx]
            try:
                def _r(nu_try):
                    t, c = 0.0, 0
                    xi_c = xi_b[len(xi_b) // 2]
                    for i, xi in enumerate(xi_b):
                        for j, N in enumerate(widths):
                            x = (xi - xi_c) * N ** (1.0 / max(nu_try, 1e-3))
                            t += (acc_b[i, j] - 1.0 / (1.0 + np.exp(-x))) ** 2
                            c += 1
                    return t / max(c, 1)
                rb = minimize_scalar(_r, bounds=(0.3, 4.0), method="bounded")
                boots.append(float(rb.x))
            except Exception:
                boots.append(nu_best)
        boots = np.array(boots)
        return nu_best, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))