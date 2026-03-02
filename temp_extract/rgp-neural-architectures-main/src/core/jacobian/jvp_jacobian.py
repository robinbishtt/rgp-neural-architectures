"""
src/core/jacobian/jvp_jacobian.py

JVPJacobian: memory-efficient forward-mode Jacobian via Jacobian-Vector Products.
Requires O(n_in) forward passes. Preferred when n_in < n_out.
"""
from __future__ import annotations
import torch


class JVPJacobian:
    """
    Forward-mode Jacobian via Jacobian-Vector Products (JVP).

    Memory:  O(n_in * n_out) per column
    Compute: O(n_in) forward passes
    Use for: wide-input, narrow-output networks (n_in < n_out).
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian dy/dx column by column.

        Each column J[:,i] = JVP(fn, x, e_i) where e_i is the i-th standard basis vector.

        Args:
            fn: callable mapping x -> y (must support functorch jvp)
            x:  input tensor

        Returns:
            J: (n_out, n_in) Jacobian matrix
        """
        n_in = x.numel()
        cols = []
        for i in range(n_in):
            v = torch.zeros_like(x)
            v.view(-1)[i] = 1.0
            _, jvp = torch.func.jvp(fn, (x,), (v,))
            cols.append(jvp.view(-1))
        return torch.stack(cols, dim=1)

    def directional_derivative(
        self, fn, x: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute the directional derivative J @ v without forming the full Jacobian."""
        _, jvp = torch.func.jvp(fn, (x,), (v,))
        return jvp
