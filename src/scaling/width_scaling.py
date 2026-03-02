"""
src/scaling/width_scaling.py

WidthScalingAnalyzer: analyzes how observables (Fisher eigenvalue density,
Jacobian spectrum, correlation length) scale with network width N.

Implements the finite-size scaling (FSS) ansatz in the width direction:
    O(N) = N^{-α/ν} · f(N^{1/ν} (g - g_c))
where g is a control parameter (e.g., σ_w²) and g_c is the critical value.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class WidthScalingResult:
    widths:           np.ndarray     # network widths N
    observables:      np.ndarray     # per-width observable values
    critical_width:   Optional[float]
    scaling_exponent: Optional[float]  # α/ν
    correlation_nu:   Optional[float]  # ν (correlation length exponent)
    r2:               float


class WidthScalingAnalyzer:
    """
    Analyzes the dependence of network observables on width N.

    In the large-N (thermodynamic) limit, observables should converge
    to their infinite-width values. The rate of convergence is governed
    by the FSS exponents, which characterize the universality class.

    The analyzer extracts:
        1. The infinite-width limit via extrapolation.
        2. The FSS exponents α/ν and ν.
        3. The data collapse quality Q.
    """

    def fit_power_law(
        self,
        widths:      np.ndarray,
        observables: np.ndarray,
    ) -> WidthScalingResult:
        """
        Fit O(N) = A · N^{-γ} + O_∞ (power-law finite-size correction).

        Args:
            widths:      array of network widths
            observables: corresponding observable measurements

        Returns:
            WidthScalingResult with fitted exponent and extrapolated limit
        """
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
        """
        Estimate the infinite-width limit O(∞) by Richardson extrapolation.

        Args:
            widths:      sorted array of network widths (ascending)
            observables: per-width observable values

        Returns:
            Extrapolated O(∞)
        """
        widths      = np.asarray(widths, float)
        observables = np.asarray(observables, float)
        # Two-point Richardson: O_∞ ≈ O(2N) + (O(2N) - O(N)) / (r^p - 1)
        # Here: fit 1/N correction
        inv_N = 1.0 / widths
        coeffs = np.polyfit(inv_N, observables, 1)
        return float(coeffs[1])  # intercept = O(∞)
 