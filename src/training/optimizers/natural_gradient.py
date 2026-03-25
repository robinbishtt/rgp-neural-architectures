from __future__ import annotations
from typing import Optional
import torch
from torch.optim import Optimizer
class DiagonalNaturalGradient(Optimizer):
    def __init__(
        self,
        params,
        lr:            float = 1e-3,
        damping:       float = 1e-4,
        n_samples:     int   = 10,
        update_freq:   int   = 10,
    ) -> None:
        defaults = dict(lr=lr, damping=damping, n_samples=n_samples,
                        update_freq=update_freq)
        super().__init__(params, defaults)
        self._step_count = 0
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "fisher_diag" not in state:
                    state["fisher_diag"] = torch.ones_like(p.data)
                if self._step_count % group["update_freq"] == 0:
                    state["fisher_diag"] = (
                        0.9 * state["fisher_diag"]
                        + 0.1 * (p.grad.data ** 2)
                    )
                damp   = group["damping"]
                F_diag = state["fisher_diag"].clamp(min=damp)
                p.data -= group["lr"] * p.grad.data / F_diag
        return loss