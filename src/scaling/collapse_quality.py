"""
src/scaling/collapse_quality.py

CollapseQualityMetrics: quantitative assessment of finite-size scaling
data collapse quality using chi-squared statistics and residual analysis.

A high-quality data collapse (Q ~ 1) validates the FSS hypothesis and
confirms the extracted scaling exponents.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


@dataclass
class CollapseQuality:
    chi_squared:       float   # reduced chi-squared of the collapse
    q_value:           float   # goodness-of-fit Q value (Q ~ 1 = good collapse)
    mean_residual:     float   # mean absolute residual from master curve
    max_residual:      float
    n_points:          int
    passed:            bool    # Q > 0.1 considered good collapse


class CollapseQualityMetrics:
    """
    Assesses the quality of finite-size scaling data collapse.

    A FSS collapse rescales data from different system sizes L onto a
    master curve f(x) where x = L^{1/ν} (g - g_c). The quality of the
    collapse is measured by the deviation of rescaled data from a smooth
    interpolating spline fit to the master curve.
    """

    def __init__(self, q_threshold: float = 0.1) -> None:
        """
        Args:
            q_threshold: minimum Q-value for a "passed" collapse.
        """
        self.q_threshold = q_threshold

    def evaluate(
        self,
        rescaled_x:  List[np.ndarray],
        rescaled_y:  List[np.ndarray],
        errors:      Optional[List[np.ndarray]] = None,
    ) -> CollapseQuality:
        """
        Evaluate collapse quality from rescaled (x, y) data.

        Args:
            rescaled_x: list of x-arrays per system size after rescaling
            rescaled_y: list of y-arrays per system size after rescaling
            errors:     optional list of y-error arrays per system size

        Returns:
            CollapseQuality with chi-squared and Q-value statistics.
        """
        x_all = np.concatenate(rescaled_x)
        y_all = np.concatenate(rescaled_y)
        e_all = (
            np.concatenate(errors) if errors
            else np.ones_like(y_all) * y_all.std()
        )

        # Fit master curve via cubic spline
        sort_idx  = np.argsort(x_all)
        x_sorted  = x_all[sort_idx]
        y_sorted  = y_all[sort_idx]
        try:
            master = interp1d(
                x_sorted, y_sorted, kind="cubic",
                fill_value="extrapolate", bounds_error=False
            )
        except Exception:
            master = interp1d(x_sorted, y_sorted, kind="linear",
                              fill_value="extrapolate", bounds_error=False)

        y_pred     = master(x_all)
        residuals  = y_all - y_pred
        chi_sq     = float(np.sum((residuals / (e_all + 1e-12)) ** 2))
        n_dof      = max(len(y_all) - 3, 1)
        red_chi_sq = chi_sq / n_dof

        from scipy.stats import chi2
        q_value = float(1.0 - chi2.cdf(chi_sq, df=n_dof))

        return CollapseQuality(
            chi_squared=red_chi_sq,
            q_value=q_value,
            mean_residual=float(np.mean(np.abs(residuals))),
            max_residual=float(np.max(np.abs(residuals))),
            n_points=len(y_all),
            passed=(q_value >= self.q_threshold),
        )
 