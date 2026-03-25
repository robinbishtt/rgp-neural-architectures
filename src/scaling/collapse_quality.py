from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.interpolate import interp1d
@dataclass
class CollapseQuality:
    chi_squared:       float   
    q_value:           float   
    mean_residual:     float   
    max_residual:      float
    n_points:          int
    passed:            bool    
class CollapseQualityMetrics:
    def __init__(self, q_threshold: float = 0.1) -> None:
        self.q_threshold = q_threshold
    def evaluate(
        self,
        rescaled_x:  List[np.ndarray],
        rescaled_y:  List[np.ndarray],
        errors:      Optional[List[np.ndarray]] = None,
    ) -> CollapseQuality:
        x_all = np.concatenate(rescaled_x)
        y_all = np.concatenate(rescaled_y)
        e_all = (
            np.concatenate(errors) if errors
            else np.ones_like(y_all) * y_all.std()
        )
        sort_idx  = np.argsort(x_all)
        x_sorted  = x_all[sort_idx]
        y_sorted  = y_all[sort_idx]
        _, unique_idx = np.unique(x_sorted, return_index=True)
        x_unique = x_sorted[unique_idx]
        y_unique = y_sorted[unique_idx]
        if len(x_unique) < 2:
            from dataclasses import fields
            return CollapseQuality(
                chi_squared=float("nan"), q_value=float("nan"),
                mean_residual=float("nan"), max_residual=float("nan"),
                n_points=len(x_all), passed=False,
            )
        try:
            kind = "cubic" if len(x_unique) >= 4 else "linear"
            master = interp1d(
                x_unique, y_unique, kind=kind,
                fill_value="extrapolate", bounds_error=False
            )
        except Exception:
            master = interp1d(x_unique, y_unique, kind="linear",
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