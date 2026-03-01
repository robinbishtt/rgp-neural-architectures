"""
src/core/lyapunov/standard_qr.py

StandardQRAlgorithm: Benettin et al. QR method for Lyapunov exponent
estimation with periodic QR re-orthogonalization of the accumulated
Jacobian product.

Reference: Benettin, G. et al. (1980). Lyapunov characteristic exponents
           for smooth dynamical systems. Meccanica, 15(1):9–30.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LyapunovResult:
    exponents:        np.ndarray   # full spectrum, descending order
    mle:              float        # maximum Lyapunov exponent
    lyapunov_sum:     float        # entropy production rate = sum of positive exponents
    kaplan_yorke_dim: float        # Kaplan-Yorke attractor dimension estimate
    regime:           str          # "ordered" | "critical" | "chaotic"


class StandardQRAlgorithm:
    """
    Benettin et al. QR algorithm for computing the Lyapunov spectrum.

    At each layer k, the product of Jacobians J_k J_{k-1} ... J_1 is
    periodically QR-decomposed to prevent numerical overflow and extract
    the growth rates (Lyapunov exponents).

    The i-th Lyapunov exponent is the average log of the i-th diagonal
    entry of R across all re-orthogonalization steps.
    """

    def __init__(
        self,
        reortho_interval: int = 10,
        n_warmup: int         = 5,
    ) -> None:
        """
        Args:
            reortho_interval: QR re-orthogonalization every N layers.
            n_warmup:         number of initial layers to discard (transient).
        """
        self.reortho_interval = reortho_interval
        self.n_warmup         = n_warmup

    def compute(
        self,
        jacobians: List[torch.Tensor],
    ) -> LyapunovResult:
        """
        Compute the Lyapunov spectrum from a sequence of layer Jacobians.

        Args:
            jacobians: list of (n_out_k, n_in_k) Jacobian matrices per layer.

        Returns:
            LyapunovResult with full spectrum and diagnostics.
        """
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
        n_pos      = np.sum(exponents > 0)
        sum_pos    = float(np.sum(exponents[exponents > 0]))

        # Kaplan-Yorke dimension
        cumsum = np.cumsum(exponents)
        j      = np.searchsorted(-cumsum, 0.0)  # first index where cumsum < 0
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
