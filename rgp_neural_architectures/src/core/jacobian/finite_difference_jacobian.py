"""
src/core/jacobian/finite_difference_jacobian.py

FiniteDifferenceJacobian: numerical Jacobian using central differences.
Used for verification of autograd-based implementations and for functions
that do not support autograd.
"""
from __future__ import annotations
from typing import Optional
import torch


class FiniteDifferenceJacobian:
    """
    Numerical Jacobian via central finite differences.

    J[:,i] ≈ [f(x + ε·eᵢ) - f(x - ε·eᵢ)] / (2ε)

    Memory:  O(n_out * n_in)
    Compute: O(2 * n_in) forward passes
    Use for: numerical verification, non-differentiable functions.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        """
        Args:
            eps: finite difference step size (default 1e-5).
                 For float32, values in [1e-5, 1e-4] are recommended.
                 For float64, values in [1e-7, 1e-6] are recommended.
        """
        self.eps = eps

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian dy/dx via central differences.

        Args:
            fn: callable mapping x -> y (does not need to be differentiable)
            x:  input tensor

        Returns:
            J: (n_out, n_in) Jacobian matrix
        """
        x = x.detach().clone()
        n_in  = x.numel()
        y0    = fn(x)
        n_out = y0.numel()
        J     = torch.zeros(n_out, n_in, dtype=x.dtype)
        x_flat = x.view(-1)
        for i in range(n_in):
            xp = x_flat.clone(); xp[i] += self.eps
            xm = x_flat.clone(); xm[i] -= self.eps
            fp = fn(xp.view(x.shape)).detach().view(-1)
            fm = fn(xm.view(x.shape)).detach().view(-1)
            J[:, i] = (fp - fm) / (2.0 * self.eps)
        return J

    def relative_error(
        self,
        J_analytic: torch.Tensor,
        J_numerical: Optional[torch.Tensor] = None,
        fn=None,
        x: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute relative Frobenius norm error between analytic and numerical Jacobians.

        Args:
            J_analytic:  analytically computed Jacobian
            J_numerical: numerically computed Jacobian (or compute from fn, x)
            fn, x:       used to compute J_numerical if not provided

        Returns:
            Relative error ||J_a - J_n||_F / ||J_n||_F
        """
        if J_numerical is None:
            if fn is None or x is None:
                raise ValueError("Either J_numerical or (fn, x) must be provided.")
            J_numerical = self.compute(fn, x)
        diff = (J_analytic - J_numerical).norm(p="fro")
        denom = J_numerical.norm(p="fro")
        return (diff / (denom + 1e-12)).item()
