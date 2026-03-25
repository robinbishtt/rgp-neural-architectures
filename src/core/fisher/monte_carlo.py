from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
class FisherMonteCarloEstimator:
    def __init__(
        self,
        n_samples: int = 100,
        estimator: str = "hutchinson",
    ) -> None:
        if estimator not in ("hutchinson", "rademacher"):
            raise ValueError(f"Unknown estimator: {estimator!r}")
        self.n_samples  = n_samples
        self.estimator  = estimator
    def _sample_vector(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randint(0, 2, shape, device=device).float() * 2.0 - 1.0
    def estimate_trace(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        device = inputs.device
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        trace_estimates = []
        for _ in range(self.n_samples):
            v = self._sample_vector(torch.Size([n_params]), device)
            model.zero_grad()
            output = model(inputs)
            loss   = loss_fn(output, targets)
            grads  = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            hv = torch.autograd.grad(
                (grad_vec * v.detach()).sum(), params,
                retain_graph=False,
            )
            hv_vec = torch.cat([h.view(-1) for h in hv])
            trace_estimates.append(float((v * hv_vec).sum()))
        return float(sum(trace_estimates) / len(trace_estimates))
    def estimate_diagonal(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        params = [p for p in model.parameters() if p.requires_grad]
        diag_accum = None
        for _ in range(self.n_samples):
            model.zero_grad()
            output = model(inputs)
            loss   = loss_fn(output, targets)
            grads  = torch.autograd.grad(loss, params, retain_graph=False)
            g_vec  = torch.cat([g.view(-1) for g in grads])
            if diag_accum is None:
                diag_accum = g_vec.detach() ** 2
            else:
                diag_accum += g_vec.detach() ** 2
        return diag_accum / self.n_samples