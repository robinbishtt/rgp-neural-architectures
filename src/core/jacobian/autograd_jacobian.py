from __future__ import annotations
import torch
class AutogradJacobian:
    """Full Jacobian via row-by-row reverse-mode automatic differentiation.

    Complexity is O(n_out) backward passes, each costing O(n_in) memory, so
    the total allocation is O(n_out × n_in).  For large square Jacobians this
    is prohibitive; prefer :class:`JVPJacobian` (O(n_in) forward passes) when
    n_in < n_out, or :class:`VJPJacobian` when n_out < n_in.

    The input tensor ``x`` is **not** mutated; a clone is used internally.
    This preserves gradient flow to upstream tensors instead of detaching.
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute the Jacobian J = ∂y/∂x ∈ ℝ^{n_out × n_in}.

        Args:
            fn: Differentiable function mapping ``x`` to a tensor ``y``.
            x:  Input tensor (any shape); treated as a flat vector of size
                ``n_in = x.numel()``.

        Returns:
            Jacobian matrix of shape ``(n_out, n_in)``.
        """
        x = x.clone().requires_grad_(True)
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
        """Return singular values σ_i of the Jacobian in descending order.

        Args:
            fn: Differentiable function.
            x:  Input tensor.

        Returns:
            1-D tensor of singular values (length ``min(n_out, n_in)``).
        """
        J = self.compute(fn, x)
        return torch.linalg.svdvals(J)
