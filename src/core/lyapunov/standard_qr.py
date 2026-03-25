from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
@dataclass
class LyapunovResult:
    exponents:        np.ndarray   
    mle:              float        
    lyapunov_sum:     float        
    kaplan_yorke_dim: float        
    regime:           str          
class StandardQRAlgorithm:
    def __init__(
        self,
        reortho_interval: int = 10,
        n_warmup: int         = 5,
    ) -> None:
        self.reortho_interval = reortho_interval
        self.n_warmup         = n_warmup
    def compute(
        self,
        jacobians: List[torch.Tensor],
    ) -> LyapunovResult:
        log_r_diags: List[np.ndarray] = []
        Q = None
        for k, J in enumerate(jacobians):
            J_np = J.detach().cpu().numpy().astype(float)
            if Q is None:
                n = J_np.shape[1]
                Q = np.eye(n, min(n, J_np.shape[0]))
            M = J_np @ Q
            if (k + 1) % self.reortho_interval == 0 and k >= self.n_warmup:
                Q, R = np.linalg.qr(M)
                diag = np.abs(np.diag(R))
                log_r_diags.append(np.log(diag + 1e-30))
            else:
                Q, _ = np.linalg.qr(M)
        if not log_r_diags:
            n = jacobians[-1].shape[-1]
            exponents = np.zeros(n)
        else:
            exponents = np.mean(log_r_diags, axis=0)
        exponents  = np.sort(exponents)[::-1]
        mle        = float(exponents[0])
        np.sum(exponents > 0)
        sum_pos    = float(np.sum(exponents[exponents > 0]))
        cumsum = np.cumsum(exponents)
        j      = np.searchsorted(-cumsum, 0.0)  
        if j == 0:
            ky_dim = 0.0
        elif j >= len(exponents):
            ky_dim = float(len(exponents))
        else:
            ky_dim = float(j) + cumsum[j - 1] / (abs(exponents[j]) + 1e-12)
        if mle < -0.05:
            regime = "ordered"
        elif mle > 0.05:
            regime = "chaotic"
        else:
            regime = "critical"
        return LyapunovResult(
            exponents=exponents,
            mle=mle,
            lyapunov_sum=sum_pos,
            kaplan_yorke_dim=ky_dim,
            regime=regime,
        )