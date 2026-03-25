from __future__ import annotations
import torch
class VJPJacobian:
    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
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
        y = fn(x)
        vjp = torch.autograd.grad(y, x, grad_outputs=v)[0]
        return vjp