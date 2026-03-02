"""
src/core/jacobian/vjp_jacobian.py

VJPJacobian: reverse-mode Jacobian via Vector-Jacobian Products.
Requires O(n_out) backward passes. Preferred when n_out < n_in.
"""
from __future__ import annotations
import torch


class VJPJacobian:
    """
    Reverse-mode Jacobian via Vector-Jacobian Products (VJP).

    Memory:  O(n_out * n_in) total
    Compute: O(n_out) backward passes
    Use for: narrow-output networks, standard neural network training (n_out < n_in).
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian dy/dx row by row.

        Each row J[i,:] = VJP(fn, x, e_i) where e_i is the i-th output basis vector.

        Args:
            fn: callable mapping x -> y
            x:  input tensor

        Returns:
            J: (n_out, n_in) Jacobian matrix
        """
        y = fn(x)
        n_out = y.numel()
        rows = []
        for i in range(n_out):
            v = torch.zeros_like(y)
            v.view(-1)[i] = 1.0
            vjp = torch.autograd.grad(
                y, x, grad_outputs=v,
                retain_graph=True,
                create_graph=False,
            )[0]
            rows.append(vjp.view(-1))
        return torch.stack(rows, dim=0)

    def gradient(self, fn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute vᵀ J without forming the full Jacobian (scalar loss gradient)."""
        y = fn(x)
        vjp = torch.autograd.grad(y, x, grad_outputs=v)[0]
        return vjp
 