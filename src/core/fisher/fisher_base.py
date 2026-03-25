from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
class FisherMetricBase(ABC):
    @abstractmethod
    def compute_layer_metric(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
    @abstractmethod
    def compute_all_layers(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> list:
    def condition_number(self, g: torch.Tensor) -> float:
        evs = torch.linalg.eigvalsh(g)
        evs = evs[evs > 0]
        if len(evs) == 0:
            return float("inf")
        return float(evs[-1] / (evs[0] + 1e-12))
    def effective_rank(self, g: torch.Tensor, threshold: float = 0.01) -> int:
        evs = torch.linalg.eigvalsh(g)
        evs = evs[evs > 0]
        if len(evs) == 0:
            return 0
        cutoff = threshold * evs[-1]
        return int((evs > cutoff).sum().item())
    def is_positive_semidefinite(
        self, g: torch.Tensor, tol: float = -1e-6
    ) -> bool:
        evs = torch.linalg.eigvalsh(g)
        return bool((evs >= tol).all().item())