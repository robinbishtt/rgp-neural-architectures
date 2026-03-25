from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
@dataclass
class LyapunovResult:
    exponents:     np.ndarray   
    mle:           float        
    lyapunov_sum:  float        
    kaplan_yorke_dim: float     
    regime:        str          
class StandardQRAlgorithm:
    def __init__(
        self,
        reortho_interval: int = 10,
        n_warmup: int = 5,
    ) -> None:
        self.reortho_interval = reortho_interval
        self.n_warmup         = n_warmup
    def compute(
        self,
        jacobians: List[np.ndarray],
        n_exponents: Optional[int] = None,
    ) -> np.ndarray:
        if not jacobians:
            return np.array([])
        n = min(J.shape[0] for J in jacobians)
        if n_exponents is None:
            n_exponents = n
        Q = np.eye(n)
        log_sv_sum = np.zeros(n_exponents)
        count = 0
        for step, J in enumerate(jacobians):
            J_sq = J[:n, :n]
            Q    = J_sq @ Q
            if (step + 1) % self.reortho_interval == 0:
                Q, R = np.linalg.qr(Q)
                diag_r = np.abs(np.diag(R))
                if step >= self.n_warmup * self.reortho_interval:
                    log_sv_sum += np.log(diag_r[:n_exponents] + 1e-12)
                    count += 1
        if count == 0:
            return np.zeros(n_exponents)
        exponents = log_sv_sum / count
        return np.sort(exponents)[::-1]
class AdaptiveQRAlgorithm(StandardQRAlgorithm):
    def __init__(
        self,
        base_interval: int = 10,
        max_condition: float = 1e6,
    ) -> None:
        super().__init__(reortho_interval=base_interval)
        self.max_condition = max_condition
    def compute(self, jacobians: List[np.ndarray], n_exponents=None) -> np.ndarray:
        if not jacobians:
            return np.array([])
        n = min(J.shape[0] for J in jacobians)
        if n_exponents is None:
            n_exponents = n
        Q = np.eye(n)
        log_sv_sum = np.zeros(n_exponents)
        count = 0
        interval = self.reortho_interval
        for step, J in enumerate(jacobians):
            J_sq = J[:n, :n]
            Q    = J_sq @ Q
            cond = np.linalg.cond(Q)
            if cond > self.max_condition:
                interval = max(1, interval // 2)
            if (step + 1) % interval == 0:
                Q, R = np.linalg.qr(Q)
                log_sv_sum += np.log(np.abs(np.diag(R))[:n_exponents] + 1e-12)
                count += 1
        if count == 0:
            return np.zeros(n_exponents)
        return np.sort(log_sv_sum / count)[::-1]
def detect_regime(
    exponents: np.ndarray,
    tolerance: float = 0.02,
) -> Literal["ordered", "critical", "chaotic"]:
    mle = float(exponents[0]) if len(exponents) > 0 else 0.0
    if mle > tolerance:
        return "chaotic"
    if abs(mle) <= tolerance:
        return "critical"
    return "ordered"
def kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    exponents = np.sort(exponents)[::-1]
    cumsum = np.cumsum(exponents)
    j = np.searchsorted(-cumsum, 0)
    if j == 0 or j >= len(exponents):
        return float(j)
    return float(j) + cumsum[j - 1] / abs(exponents[j] + 1e-12)
def analyze_lyapunov(jacobians: List[np.ndarray]) -> LyapunovResult:
    algo = AdaptiveQRAlgorithm()
    exponents = algo.compute(jacobians)
    regime    = detect_regime(exponents)
    ky_dim    = kaplan_yorke_dimension(exponents)
    return LyapunovResult(
        exponents=exponents,
        mle=float(exponents[0]) if len(exponents) else 0.0,
        lyapunov_sum=float(exponents.sum()),
        kaplan_yorke_dim=ky_dim,
        regime=regime,
    )