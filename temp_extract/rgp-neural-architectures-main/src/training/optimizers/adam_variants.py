"""
src/training/optimizers/adam_variants.py

Adam-family optimizers: standard Adam, AMSGrad, AdaBound variants.
"""
from __future__ import annotations
import torch
from torch.optim import Adam


def build_adam(params, lr: float = 1e-3, weight_decay: float = 1e-4,
               amsgrad: bool = False) -> Adam:
    """Standard Adam or AMSGrad variant."""
    return Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)


class AdaBound(torch.optim.Optimizer):
    """
    AdaBound: adaptive gradient clipping that transitions to SGD.
    Combines Adam's fast convergence with SGD's good generalization.

    From: Luo et al. (2019) "Adaptive Gradient Methods with Dynamic Bound
    of Learning Rate"
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 final_lr: float = 0.1, gamma: float = 1e-3,
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr,
                        gamma=gamma, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.base_lrs = [g["lr"] for g in self.param_groups]

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"]  = 0
                    state["exp_avg"]   = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                b1, b2 = group["betas"]
                state["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                step_size = group["lr"] / bc1
                denom = (state["exp_avg_sq"].sqrt() / (bc2 ** 0.5)).add_(group["eps"])
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size_clipped = torch.clamp(step_size / denom, lower, upper)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                p.data.addcmul_(state["exp_avg"] / bc1, step_size_clipped, value=-1)
        return loss
