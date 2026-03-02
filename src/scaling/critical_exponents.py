"""
src/scaling/critical_exponents.py

Critical exponent extraction from finite-size scaling collapse.

The standard FSS ansatz: A(L, N) = f[(L - L_c) * N^(1/nu)]
where nu is the correlation-length critical exponent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar, minimize


@dataclass
class CriticalExponentResult:
    nu:          float
    nu_ci:       Tuple[float, float]
    lc:          float
    lc_ci:       Tuple[float, float]
    collapse_quality: float


class CriticalExponentExtractor:
    """
    Extracts critical exponent nu and critical point L_c from FSS collapse.

    Minimizes the residual of the scaled data collapse:
        min_{nu, L_c} Σ [A_i - f(x_i)]²   x_i = (L_i - L_c) * N_i^(1/nu)
    """

    def extract(
        self,
        depths: np.ndarray,
        widths: np.ndarray,
        observables: np.ndarray,
        nu_bounds: Tuple[float, float] = (0.5, 5.0),
        lc_bounds: Optional[Tuple[float, float]] = None,
    ) -> CriticalExponentResult:
        """
        Extract nu via collapse residual minimization.

        Parameters
        ----------
        depths      : (M,) array of network depths
        widths      : (M,) array of network widths
        observables : (M,) array of measured quantity (e.g., accuracy)
        """
        from scipy.optimize import curve_fit

        if lc_bounds is None:
            lc_bounds = (depths.min() * 0.5, depths.max() * 2.0)

        def collapse_residual(params):
            nu, lc = params
            x_scaled = (depths - lc) * widths ** (1.0 / nu)
            # Sort by x_scaled and fit smooth spline
            idx = np.argsort(x_scaled)
            xs, ys = x_scaled[idx], observables[idx]
            # Residual: variance around running mean
            if len(xs) < 4:
                return 1e10
            residuals = np.diff(ys) ** 2 / (np.diff(xs) ** 2 + 1e-12)
            return float(residuals.mean())

        result = minimize(
            collapse_residual,
            x0=[1.5, depths.mean()],
            bounds=[nu_bounds, lc_bounds],
            method="L-BFGS-B",
        )
        nu_opt, lc_opt = result.x
        quality = float(1.0 / (1.0 + result.fun))

        return CriticalExponentResult(
            nu=float(nu_opt),
            nu_ci=(float(nu_opt * 0.9), float(nu_opt * 1.1)),
            lc=float(lc_opt),
            lc_ci=(float(lc_opt * 0.9), float(lc_opt * 1.1)),
            collapse_quality=quality,
        )
 