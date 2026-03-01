"""
src/scaling/data_collapse.py

Tests quality of FSS data collapse via chi-squared and R² metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CollapseQuality:
    r2:           float
    chi_squared:  float
    n_dof:        int
    passed:       bool


class DataCollapseVerifier:
    """
    Verifies quality of finite-size scaling data collapse.

    After optimal nu and L_c are found, tests whether the collapsed
    data (L-L_c)*N^{1/nu} vs observable falls onto a single master curve.
    """

    def __init__(self, r2_threshold: float = 0.95) -> None:
        self.r2_threshold = r2_threshold

    def verify(
        self,
        x_scaled: np.ndarray,
        y_obs: np.ndarray,
    ) -> CollapseQuality:
        """
        Fit y = f(x_scaled) with cubic spline and compute R².

        Parameters
        ----------
        x_scaled : scaled variable (L - L_c) * N^(1/nu)
        y_obs    : observed quantity
        """
        from scipy.interpolate import UnivariateSpline
        idx     = np.argsort(x_scaled)
        xs, ys  = x_scaled[idx], y_obs[idx]

        try:
            spline  = UnivariateSpline(xs, ys, k=3, s=len(ys) * 0.1)
            y_pred  = spline(xs)
            ss_res  = ((ys - y_pred) ** 2).sum()
            ss_tot  = ((ys - ys.mean()) ** 2).sum()
            r2      = float(1.0 - ss_res / max(ss_tot, 1e-12))
            chi2    = float(ss_res / max(len(ys) - 4, 1))
        except Exception:
            r2   = 0.0
            chi2 = 1e10

        return CollapseQuality(
            r2=r2,
            chi_squared=chi2,
            n_dof=max(len(ys) - 4, 1),
            passed=r2 >= self.r2_threshold,
        )
