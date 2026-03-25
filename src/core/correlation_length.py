from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit
@dataclass
class CorrelationLengthResult:
    xi_values:  np.ndarray   
    xi_0:       float        
    k_c:        float        
    r2:         float        
    chi1:       float        
    xi_0_ci:    Tuple[float, float] = (0.0, 0.0)
    k_c_ci:     Tuple[float, float] = (0.0, 0.0)
def _exp_decay(k: np.ndarray, xi_0: float, k_c: float) -> np.ndarray:
    return xi_0 * np.exp(-k / k_c)
class FisherSpectrumMethod:
    def estimate(self, eigenvalue_lists: list) -> np.ndarray:
        xi_values = []
        for ev in eigenvalue_lists:
            ev = np.asarray(ev)
            ev = ev[ev > 1e-10]
            if len(ev) == 0:
                xi_values.append(0.0)
            else:
                inv_mean = np.mean(1.0 / ev)
                xi_values.append(float(1.0 / np.sqrt(inv_mean + 1e-12)))
        return np.array(xi_values)
class ExponentialDecayFitter:
    def fit(self, xi_values: np.ndarray) -> CorrelationLengthResult:
        k = np.arange(len(xi_values), dtype=float)
        xi = np.asarray(xi_values, dtype=float)
        try:
            popt, pcov = curve_fit(
                _exp_decay, k, xi,
                p0=[xi[0], max(len(xi) / 3.0, 1.0)],
                bounds=([0.0, 0.1], [np.inf, np.inf]),
                maxfev=10000,
            )
            xi_0, k_c = popt
            perr = np.sqrt(np.diag(pcov))
            xi_0_ci = (xi_0 - 2 * perr[0], xi_0 + 2 * perr[0])
            k_c_ci  = (k_c  - 2 * perr[1], k_c  + 2 * perr[1])
        except RuntimeError:
            xi_0, k_c = xi[0], float(len(xi))
            xi_0_ci = (0.0, 0.0)
            k_c_ci  = (0.0, 0.0)
        xi_pred  = _exp_decay(k, xi_0, k_c)
        ss_res   = ((xi - xi_pred) ** 2).sum()
        ss_tot   = ((xi - xi.mean()) ** 2).sum()
        r2       = 1.0 - ss_res / max(ss_tot, 1e-12)
        chi1     = float(np.exp(-1.0 / k_c)) if k_c > 0 else 0.0
        return CorrelationLengthResult(
            xi_values=xi,
            xi_0=float(xi_0),
            k_c=float(k_c),
            r2=float(r2),
            chi1=chi1,
            xi_0_ci=xi_0_ci,
            k_c_ci=k_c_ci,
        )
class MaximumLikelihoodEstimator:
    def fit(self, xi_values: np.ndarray) -> CorrelationLengthResult:
        k   = np.arange(len(xi_values), dtype=float)
        xi  = np.asarray(xi_values, dtype=float)
        log_xi = np.log(xi + 1e-12)
        A    = np.vstack([np.ones_like(k), -k]).T
        coef, _, _, _ = np.linalg.lstsq(A, log_xi, rcond=None)
        log_xi0, inv_kc = coef
        xi_0  = float(np.exp(log_xi0))
        k_c   = float(1.0 / max(inv_kc, 1e-6))
        chi1  = float(np.exp(-1.0 / k_c))
        xi_pred = _exp_decay(k, xi_0, k_c)
        ss_res  = ((xi - xi_pred) ** 2).sum()
        ss_tot  = ((xi - xi.mean()) ** 2).sum()
        r2      = 1.0 - ss_res / max(ss_tot, 1e-12)
        return CorrelationLengthResult(
            xi_values=xi, xi_0=xi_0, k_c=k_c, r2=float(r2), chi1=chi1,
        )
class TransferMatrixMethod:
    def estimate(self, transfer_matrices: list) -> np.ndarray:
        xi_values = []
        for T in transfer_matrices:
            T  = np.asarray(T)
            ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
            if len(ev) >= 2 and ev[0] > 0 and ev[1] > 0:
                ratio = ev[1] / ev[0]
                if ratio < 1.0 and ratio > 0:
                    xi = -1.0 / np.log(ratio)
                    xi_values.append(float(xi))
                else:
                    xi_values.append(float(ev[0]))
            else:
                xi_values.append(1.0)
        return np.array(xi_values)