from __future__ import annotations
import torch
class JVPJacobian:
    def compute(self, fn, x: torch.Tensor) -> torch.Tensor:
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
        _, jvp = torch.func.jvp(fn, (x,), (v,))
        return jvp