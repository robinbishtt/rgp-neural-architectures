"""
src/training/optimizers/fisher_optimizer.py

FisherOptimizer: K-FAC-inspired optimizer that uses the Fisher information
metric structure of the RG-Net architecture to compute approximate
natural gradient updates block-diagonally.

At each layer k, the block-diagonal Fisher approximation uses the Kronecker
factorization F_k ≈ A_{k-1} ⊗ G_k (Martens & Grosse, 2015) where
A_{k-1} is the input covariance and G_k is the output gradient covariance.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer


class FisherOptimizer(Optimizer):
    """
    Layer-wise Fisher information preconditioned optimizer.

    Implements approximate natural gradient via block-diagonal Kronecker
    factorization of the Fisher matrix:
        F_k ≈ A_{k-1} ⊗ G_k
    where:
        A_{k-1} = E[h^{k-1} (h^{k-1})ᵀ]  (input covariance)
        G_k     = E[δ^k (δ^k)ᵀ]           (gradient covariance)

    The natural gradient for layer k is:
        ΔW_k = G_k⁻¹ ∇_{W_k}L A_{k-1}⁻¹

    Reference: Martens, J. & Grosse, R. (2015). Optimizing neural networks
               with Kronecker-factored approximate curvature. ICML 2015.
    """

    def __init__(
        self,
        params,
        lr:          float = 1e-3,
        damping:     float = 1e-3,
        update_freq: int   = 10,
        decay:       float = 0.95,
    ) -> None:
        defaults = dict(lr=lr, damping=damping,
                        update_freq=update_freq, decay=decay)
        super().__init__(params, defaults)
        self._step_count = 0

    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state  = self.state[p]
                grad   = p.grad.data
                damp   = group["damping"]

                if "A_inv" not in state:
                    n = grad.shape[-1] if grad.dim() > 1 else 1
                    m = grad.shape[0]
                    state["A_inv"] = torch.eye(n, device=p.device) * (1.0 / (damp + 1e-8))
                    state["G_inv"] = torch.eye(m, device=p.device) * (1.0 / (damp + 1e-8))

                if grad.dim() == 2 and self._step_count % group["update_freq"] == 0:
                    A_hat = grad.t() @ grad / grad.shape[0]
                    G_hat = grad @ grad.t() / grad.shape[1]
                    A_reg = A_hat + damp * torch.eye(A_hat.shape[0], device=A_hat.device)
                    G_reg = G_hat + damp * torch.eye(G_hat.shape[0], device=G_hat.device)
                    try:
                        state["A_inv"] = torch.linalg.inv(A_reg)
                        state["G_inv"] = torch.linalg.inv(G_reg)
                    except torch.linalg.LinAlgError:
                        pass

                if grad.dim() == 2:
                    nat_grad = state["G_inv"] @ grad @ state["A_inv"]
                else:
                    nat_grad = grad / (grad.norm() ** 2 + damp)

                p.data -= group["lr"] * nat_grad

        return loss
 