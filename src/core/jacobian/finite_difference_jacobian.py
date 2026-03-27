from __future__ import annotations
from typing import Optional
import torch
class FiniteDifferenceJacobian:
    """Jacobian approximation via central finite differences.

    Uses the central-difference formula::

        J[:, i] ≈ (f(x + ε eᵢ) - f(x - ε eᵢ)) / (2ε)

    which is second-order accurate: the truncation error is O(ε²).  The
    output tensor is allocated on the same device as ``x``.

    Args:
        eps: Perturbation magnitude ε.  Default 1e-5 balances truncation and
             round-off errors for float32 arithmetic.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J ≈ ∂y/∂x via central finite differences.

        Args:
            fn: Callable mapping x → y (no gradient required).
            x:  Input tensor of any shape.

        Returns:
            Jacobian of shape ``(n_out, n_in)`` on the same device as ``x``.
        """
        x = x.detach().clone()
        n_in  = x.numel()
        y0    = fn(x)
        n_out = y0.numel()
        J     = torch.zeros(n_out, n_in, dtype=x.dtype, device=x.device)
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
        """Frobenius-norm relative error ‖J_analytic - J_numerical‖_F / ‖J_numerical‖_F.

        Args:
            J_analytic:  Analytically computed Jacobian.
            J_numerical: Pre-computed numerical Jacobian.  If ``None``, both
                         ``fn`` and ``x`` must be provided and the numerical
                         Jacobian is computed on the fly.
            fn:          Function to differentiate (used if ``J_numerical`` is
                         ``None``).
            x:           Evaluation point (used if ``J_numerical`` is ``None``).

        Returns:
            Scalar relative error.
        """
        if J_numerical is None:
            if fn is None or x is None:
                raise ValueError("Either J_numerical or (fn, x) must be provided.")
            J_numerical = self.compute(fn, x)
        diff = (J_analytic - J_numerical).norm(p="fro")
        denom = J_numerical.norm(p="fro")
        return (diff / (denom + 1e-12)).item()