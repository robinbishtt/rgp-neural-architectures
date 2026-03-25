from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
@dataclass
class EffectiveDimensionResult:
    participation_ratio: float
    spectral_entropy:    float
    rank_threshold:      int
    n_total:             int
    compression_ratio:   float
class FisherEffectiveDimension:
    def __init__(self, rank_threshold_eps: float = 1e-3) -> None:
        self.eps = rank_threshold_eps
    def compute(self, G: torch.Tensor) -> EffectiveDimensionResult:
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        n  = len(ev)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        p_norm  = ev / ev.sum()
        entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
        threshold = self.eps * ev.max()
        rank_t    = int((ev > threshold).sum())
        return EffectiveDimensionResult(
            participation_ratio=pr,
            spectral_entropy=entropy,
            rank_threshold=rank_t,
            n_total=n,
            compression_ratio=float(pr / n),
        )
    def layer_profile(
        self, fisher_matrices: list
    ) -> list:
        return [self.compute(G) for G in fisher_matrices]