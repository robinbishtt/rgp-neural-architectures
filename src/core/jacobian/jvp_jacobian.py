from __future__ import annotations
import torch
class JVPJacobian:
    """Full Jacobian via column-by-column forward-mode (JVP) differentiation.

    Executes n_in forward passes using ``torch.func.jvp``, collecting one
    column J[:, i] = ∂y/∂x_i per pass.  Complexity is O(n_in) JVP calls.
    This is more memory-efficient than the reverse-mode approach when n_in
    is small relative to n_out because no computation graph is retained.

    Prefer over :class:`VJPJacobian` when the function has more outputs than
    inputs (n_out > n_in).
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J = ∂y/∂x ∈ ℝ^{n_out × n_in} via forward-mode AD.

        Args:
            fn: Callable mapping a tensor of shape ``x.shape`` to an output
                tensor.  Must be compatible with ``torch.func.jvp``.
            x:  Input tensor of any shape; treated as a flat vector of size
                n_in = x.numel().

        Returns:
            Jacobian matrix of shape ``(n_out, n_in)`` assembled column by
            column from n_in JVP evaluations with standard basis vectors.
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
        """Compute the directional derivative J v = d/dt f(x + tv)|_{t=0}.

        Args:
            fn: Callable mapping x → y.
            x:  Evaluation point.
            v:  Tangent vector (same shape as x).

        Returns:
            Directional derivative of shape ``y.shape``.
        """
        _, jvp = torch.func.jvp(fn, (x,), (v,))
        return jvp