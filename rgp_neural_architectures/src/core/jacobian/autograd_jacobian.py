"""
src/core/jacobian/autograd_jacobian.py

AutogradJacobian: full Jacobian computation via repeated backward passes
using torch.autograd.grad. Exact but O(n_out) backward passes.
"""
from __future__ import annotations
import torch


class AutogradJacobian:
    """
    Full Jacobian via repeated backward passes.

    Memory:  O(n_out * n_in)
    Compute: O(n_out) backward passes
    Use for: small networks, ground-truth verification.
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian dy/dx.

        Args:
            fn: callable mapping x -> y
            x:  input tensor (any shape), will be flattened

        Returns:
            J: (n_out, n_in) Jacobian matrix
        """
        x = x.requires_grad_(True)
        y = fn(x)
        n_in  = x.numel()
        n_out = y.numel()
        J = torch.zeros(n_out, n_in, dtype=x.dtype, device=x.device)
        for i in range(n_out):
            grad = torch.autograd.grad(
                y.view(-1)[i], x,
                retain_graph=True,
                create_graph=False,
            )[0]
            J[i] = grad.view(-1)
        return J

    def singular_values(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute singular values of the Jacobian."""
        J = self.compute(fn, x)
        return torch.linalg.svdvals(J)
