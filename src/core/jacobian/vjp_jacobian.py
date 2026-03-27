from __future__ import annotations
import torch
class VJPJacobian:
    """Full Jacobian via row-by-row reverse-mode (VJP) differentiation.

    Executes n_out backward passes, each computing one row J[i, :] = eᵢᵀ ∂y/∂x
    via ``torch.autograd.grad`` with a standard basis co-vector ``eᵢ``.
    Complexity is O(n_out) backward passes.

    Prefer over :class:`JVPJacobian` when the function has fewer outputs than
    inputs (n_out < n_in) since each pass costs O(n_in) in memory but only
    n_out passes are needed.
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J = ∂y/∂x ∈ ℝ^{n_out × n_in} via reverse-mode AD.

        Args:
            fn: Callable mapping x → y.  ``x`` must be leaf with
                ``requires_grad=True``, or the function must create a
                differentiable graph with respect to ``x``.
            x:  Input tensor of any shape; treated as a flat vector of size
                n_in = x.numel().

        Returns:
            Jacobian matrix of shape ``(n_out, n_in)`` assembled row by row
            from n_out VJP evaluations with standard basis co-vectors.
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
        """Compute the vector-Jacobian product vᵀ J = vᵀ (∂y/∂x).

        Args:
            fn: Callable mapping x → y.
            x:  Evaluation point (must admit gradients).
            v:  Co-vector of shape ``y.shape``.

        Returns:
            VJP of shape ``x.shape``.
        """
        y = fn(x)
        vjp = torch.autograd.grad(y, x, grad_outputs=v)[0]
        return vjp