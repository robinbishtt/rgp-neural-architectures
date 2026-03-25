from __future__ import annotations
from typing import List
import numpy as np
import torch
from src.core.lyapunov.standard_qr import LyapunovResult, StandardQRAlgorithm
class AdaptiveQRAlgorithm(StandardQRAlgorithm):
    def __init__(
        self,
        reortho_interval:   int   = 10,
        n_warmup:           int   = 5,
        condition_threshold: float = 1e8,
        min_interval:       int   = 2,
    ) -> None:
        super().__init__(reortho_interval=reortho_interval, n_warmup=n_warmup)
        self.condition_threshold = condition_threshold
        self.min_interval        = min_interval
    def compute(
        self,
        jacobians: List[torch.Tensor],
    ) -> LyapunovResult:
        log_r_diags: list = []
        Q            = None
        last_ortho   = 0
        for k, J in enumerate(jacobians):
            J_np = J.detach().cpu().numpy().astype(float)
            if Q is None:
                n = J_np.shape[1]
                Q = np.eye(n, min(n, J_np.shape[0]))
            M = J_np @ Q
            scheduled    = (k + 1) % self.reortho_interval == 0
            gap_ok       = (k - last_ortho) >= self.min_interval
            try:
                svs = np.linalg.svd(M, compute_uv=False)
                cond = float(svs[0] / (svs[-1] + 1e-30))
                adaptive = (cond > self.condition_threshold) and gap_ok
            except np.linalg.LinAlgError:
                adaptive = True
            if (scheduled or adaptive) and k >= self.n_warmup:
                Q, R    = np.linalg.qr(M)
                diag    = np.abs(np.diag(R))
                log_r_diags.append(np.log(diag + 1e-30))
                last_ortho = k
            else:
                Q, _ = np.linalg.qr(M)
        if not log_r_diags:
            n = jacobians[-1].shape[-1]
            exponents = np.zeros(n)
        else:
            exponents = np.mean(log_r_diags, axis=0)
        exponents = np.sort(exponents)[::-1]
        mle       = float(exponents[0])
        cumsum    = np.cumsum(exponents)
        j         = np.searchsorted(-cumsum, 0.0)
        if j == 0:
            ky_dim = 0.0
        elif j >= len(exponents):
            ky_dim = float(len(exponents))
        else:
            ky_dim = float(j) + cumsum[j - 1] / (abs(exponents[j]) + 1e-12)
        regime = "ordered" if mle < -0.05 else ("chaotic" if mle > 0.05 else "critical")
        return LyapunovResult(
            exponents=exponents,
            mle=mle,
            lyapunov_sum=float(np.sum(exponents[exponents > 0])),
            kaplan_yorke_dim=ky_dim,
            regime=regime,
        )