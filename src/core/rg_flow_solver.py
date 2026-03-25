from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
@dataclass
class RGFlowSolution:
    depths:          np.ndarray    
    xi_profile:      np.ndarray    
    chi_profile:     np.ndarray    
    L_min:           float         
    xi_depth:        float         
    accuracy_curve:  np.ndarray    
    xi_0:            float         
    xi_target:       float         
class RGFlowSolver:
    def __init__(
        self,
        chi_infty:   float = 0.894,     
        sigma_w:     float = 1.4,
        sigma_b:     float = 0.3,
        width:       int   = 512,       
        include_finite_width: bool = True,
    ) -> None:
        self.chi_infty = chi_infty
        self.sigma_w   = sigma_w
        self.sigma_b   = sigma_b
        self.N         = width
        self.include_fw = include_finite_width
        if chi_infty < 1.0 and chi_infty > 0:
            self.k_c = float(-1.0 / np.log(chi_infty))
        elif chi_infty >= 1.0:
            warnings.warn(
                f"chi_infty={chi_infty:.4f} >= 1 (chaotic or critical phase). "
                f"Setting k_c = inf (no exponential decay).",
                stacklevel=2,
            )
            self.k_c = float("inf")
        else:
            raise ValueError(f"chi_infty must be positive, got {chi_infty}")
    def beta_function(self, xi: float, ell: float) -> float:
        if self.k_c == float("inf"):
            return 0.0
        leading = -xi / self.k_c
        if not self.include_fw or self.N == 0:
            return leading
        c1      = (self.sigma_w ** 2) / 2.0
        finite  = c1 / (self.N * max(xi, 1e-6))
        return leading + finite
    def solve(
        self,
        xi_0:      float,
        xi_target: float,
        L_max:     int = 500,
        n_points:  int = 501,
    ) -> RGFlowSolution:
        t_eval = np.linspace(0, L_max, n_points)
        def _rhs(t, y):
            return [self.beta_function(max(y[0], 1e-8), t)]
        sol = solve_ivp(
            _rhs,
            t_span=(0, L_max),
            y0=[xi_0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        xi_profile = np.clip(sol.y[0], 1e-8, None)
        if self.k_c < float("inf"):
            chi_profile = np.exp(-1.0 / self.k_c) * np.ones(len(xi_profile))
        else:
            chi_profile = np.ones(len(xi_profile))
        L_min = self._find_L_min(sol.t, xi_profile, xi_target)
        P_min, P_max = 0.10, 0.99  
        if self.k_c < float("inf") and not np.isnan(L_min):
            crossover_scale = self.k_c
            logit = (sol.t - L_min) / max(crossover_scale, 0.5)
            accuracy = P_min + (P_max - P_min) / (1.0 + np.exp(-logit))
        else:
            accuracy = np.full(len(sol.t), P_min)
        return RGFlowSolution(
            depths         = sol.t,
            xi_profile     = xi_profile,
            chi_profile    = chi_profile,
            L_min          = float(L_min),
            xi_depth       = float(self.k_c),
            accuracy_curve = accuracy,
            xi_0           = xi_0,
            xi_target      = xi_target,
        )
    def _find_L_min(
        self,
        depths:     np.ndarray,
        xi_profile: np.ndarray,
        xi_target:  float,
    ) -> float:
        if xi_profile[-1] > xi_target:
            return float(self.k_c * np.log(max(xi_profile[0] / xi_target, 1.01)))
        for i in range(1, len(xi_profile)):
            if xi_profile[i] <= xi_target and xi_profile[i - 1] > xi_target:
                frac = (xi_profile[i - 1] - xi_target) / (
                    xi_profile[i - 1] - xi_profile[i] + 1e-12
                )
                return float(depths[i - 1] + frac * (depths[i] - depths[i - 1]))
        return float(depths[-1])
class BetaFunctionSolver:
    def __init__(self, chi_infty: float = 0.894) -> None:
        self.chi     = chi_infty
        self.eps_0   = max(1.0 - chi_infty, 0.0)  
    def metric_contraction_profile(
        self,
        eta_0: float,
        depths: np.ndarray,
    ) -> np.ndarray:
        return eta_0 * (self.chi ** depths)
    def verify_contraction(
        self,
        eta_measured: np.ndarray,
        depths:       np.ndarray,
        eta_0:        Optional[float] = None,
        rtol:         float           = 0.10,
    ) -> Tuple[bool, float, float]:
        if eta_0 is None:
            eta_0 = float(eta_measured[0])
        eta_theory = self.metric_contraction_profile(eta_0, depths)
        ss_res = np.sum((eta_measured - eta_theory) ** 2)
        ss_tot = np.sum((eta_measured - eta_measured.mean()) ** 2)
        r2     = float(1.0 - ss_res / max(ss_tot, 1e-12))
        violations = eta_measured > eta_theory * (1.0 + rtol)
        max_viol   = float(np.max(eta_measured / (eta_theory + 1e-12)) - 1.0)
        passes     = bool(~violations.any())
        return passes, r2, max_viol