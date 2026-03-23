"""
src/core/jacobian.py

Jacobian computation strategies:
  - AutogradJacobian        : via torch.autograd
  - JVPJacobian             : forward-mode via jvp
  - VJPJacobian             : reverse-mode via vjp
  - FiniteDifferenceJacobian: numerical verification
  - CumulativeJacobian      : log singular values across depth
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class AutogradJacobian:
    """Full Jacobian via repeated backward passes."""

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        x    = x.requires_grad_(True)
        y    = fn(x)
        n_in = x.numel()
        n_out= y.numel()
        J    = torch.zeros(n_out, n_in, dtype=x.dtype, device=x.device)
        for i in range(n_out):
            grad = torch.autograd.grad(
                y.view(-1)[i], x,
                retain_graph=True, create_graph=False,
            )[0]
            J[i] = grad.view(-1)
        return J


class JVPJacobian:
    """Memory-efficient forward-mode Jacobian via Jacobian-Vector Products."""

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        n_in = x.numel()
        cols = []
        for i in range(n_in):
            v    = torch.zeros_like(x)
            v.view(-1)[i] = 1.0
            _, jvp = torch.func.jvp(fn, (x,), (v,))
            cols.append(jvp.view(-1))
        return torch.stack(cols, dim=1)


class VJPJacobian:
    """Reverse-mode Jacobian via Vector-Jacobian Products."""

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        y     = fn(x)
        n_out = y.numel()
        rows  = []
        for i in range(n_out):
            v    = torch.zeros_like(y)
            v.view(-1)[i] = 1.0
            vjp  = torch.autograd.grad(y, x, grad_outputs=v,
                                       retain_graph=True, create_graph=False)[0]
            rows.append(vjp.view(-1))
        return torch.stack(rows, dim=0)


class FiniteDifferenceJacobian:
    """Numerical Jacobian for verification. Uses central differences."""

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        x     = x.detach().clone()
        n_in  = x.numel()
        y0    = fn(x)
        n_out = y0.numel()
        J     = torch.zeros(n_out, n_in, dtype=x.dtype)
        x_flat= x.view(-1)
        for i in range(n_in):
            xp = x_flat.clone(); xp[i] += self.eps
            xm = x_flat.clone(); xm[i] -= self.eps
            fp = fn(xp.view(x.shape)).detach().view(-1)
            fm = fn(xm.view(x.shape)).detach().view(-1)
            J[:, i] = (fp - fm) / (2.0 * self.eps)
        return J


class CumulativeJacobian:
    """
    Computes log singular values of cumulative Jacobians across depth.
    Used for Lyapunov exponent estimation.
    """

    def log_singular_values(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_layers: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns array of shape (n_layers, min(N_in, N_out)) containing
        log singular values of the cumulative Jacobian at each depth.
        """
        log_sv_per_layer = []
        handles  = []
        jacobians= []

        def _hook(module, inp, out):
            # Approximate layer Jacobian via activation gradient
            if inp[0].requires_grad:
                def _save(grad):
                    jacobians.append(grad.detach())
                out.register_hook(_save)

        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(_hook))

        x = x.requires_grad_(True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for h in handles:
            h.remove()

        # Build cumulative product and extract log SVs
        cum_J = None
        for J in jacobians:
            J_2d = J.view(J.shape[0], -1)
            if cum_J is None:
                cum_J = J_2d
            else:
                min_dim = min(cum_J.shape[1], J_2d.shape[0])
                cum_J   = J_2d[:, :min_dim] @ cum_J[:min_dim, :]
            sv = torch.linalg.svdvals(cum_J).cpu().numpy()
            log_sv_per_layer.append(np.log(sv + 1e-12))

            if max_layers and len(log_sv_per_layer) >= max_layers:
                break

        return np.array(log_sv_per_layer)
 