from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
class AutogradJacobian:
    """Full Jacobian via row-by-row reverse-mode automatic differentiation.

    Complexity is O(n_out) backward passes.  The input tensor ``x`` is **not**
    mutated; a detached clone with ``requires_grad=True`` is used internally.
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J = ∂y/∂x ∈ ℝ^{n_out × n_in}.

        Args:
            fn: Callable mapping x → y (both tensors).
            x:  Input tensor of any shape; treated as a flat vector.

        Returns:
            Jacobian of shape ``(n_out, n_in)``.
        """
        x    = x.clone().requires_grad_(True)
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
    """Full Jacobian via column-by-column forward-mode (JVP) differentiation.

    Requires O(n_in) JVP calls each of O(1) extra memory, making this
    O(n_in × cost_of_fn) total.  Prefer over :class:`AutogradJacobian` when
    n_in < n_out.  Uses ``torch.func.jvp`` (functorch-style functional API).
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J = ∂y/∂x ∈ ℝ^{n_out × n_in} via forward-mode AD.

        Args:
            fn: Callable mapping x → y.
            x:  Input tensor; treated as a flat vector of size n_in.

        Returns:
            Jacobian of shape ``(n_out, n_in)`` assembled column by column.
        """
        n_in = x.numel()
        cols = []
        for i in range(n_in):
            v    = torch.zeros_like(x)
            v.view(-1)[i] = 1.0
            _, jvp = torch.func.jvp(fn, (x,), (v,))
            cols.append(jvp.view(-1))
        return torch.stack(cols, dim=1)
class VJPJacobian:
    """Full Jacobian via row-by-row reverse-mode (VJP) differentiation.

    Requires O(n_out) VJP calls.  Equivalent to :class:`AutogradJacobian` in
    complexity but uses explicit ``grad_outputs`` rather than scalar selection.
    Prefer over :class:`JVPJacobian` when n_out < n_in.
    """

    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J = ∂y/∂x ∈ ℝ^{n_out × n_in} via reverse-mode AD.

        Args:
            fn: Callable mapping x → y.  ``x`` must support gradients.
            x:  Input tensor; treated as a flat vector of size n_in.

        Returns:
            Jacobian of shape ``(n_out, n_in)`` assembled row by row.
        """
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
    """Jacobian approximation via central finite differences.

    Uses the formula J[:, i] ≈ (f(x + ε eᵢ) - f(x - ε eᵢ)) / (2ε) which
    has second-order accuracy O(ε²).  The output tensor is allocated on the
    same device as the input ``x``.

    Args:
        eps: Perturbation step size ε.  Default 1e-5 balances truncation and
             round-off errors for float32.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Compute J ≈ ∂y/∂x via central differences.

        Args:
            fn: Function mapping x → y (no gradient required).
            x:  Input tensor of any shape.

        Returns:
            Jacobian of shape ``(n_out, n_in)`` on the same device as ``x``.
        """
        x     = x.detach().clone()
        n_in  = x.numel()
        y0    = fn(x)
        n_out = y0.numel()
        J     = torch.zeros(n_out, n_in, dtype=x.dtype, device=x.device)
        x_flat= x.view(-1)
        for i in range(n_in):
            xp = x_flat.clone(); xp[i] += self.eps
            xm = x_flat.clone(); xm[i] -= self.eps
            fp = fn(xp.view(x.shape)).detach().view(-1)
            fm = fn(xm.view(x.shape)).detach().view(-1)
            J[:, i] = (fp - fm) / (2.0 * self.eps)
        return J
class CumulativeJacobian:
    """Track the cumulative product of layer Jacobians through a model.

    At each linear layer ℓ the cumulative Jacobian is J^(ℓ) = J_ℓ J_{ℓ-1} …
    J_1, and its singular values give the Lyapunov exponents of the RG flow.
    The log singular values grow (or shrink) linearly with depth at rate
    log χ_i, where χ_i are the Lyapunov exponents.
    """

    def log_singular_values(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_layers: Optional[int] = None,
    ) -> np.ndarray:
        """Return log singular values of the cumulative Jacobian per layer.

        Args:
            model:      PyTorch module whose ``nn.Linear`` layers define the
                        Jacobians.  Only layers whose input has
                        ``requires_grad=True`` are processed.
            x:          Input tensor.  A fresh computation graph is created
                        internally.
            max_layers: If set, stop after this many layers.

        Returns:
            Array of shape ``(n_layers, min(d_out, d_in))`` containing
            ``log(σ_i + 1e-12)`` for each cumulative Jacobian.
        """
        log_sv_per_layer = []
        handles  = []
        jacobians= []
        def _hook(module, inp, out):
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
