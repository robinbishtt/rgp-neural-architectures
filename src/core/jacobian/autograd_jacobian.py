from __future__ import annotations
import torch
class AutogradJacobian:
    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
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
        J = self.compute(fn, x)
        return torch.linalg.svdvals(J)