from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
@dataclass
class EigenvalueAnalysisResult:
    eigenvalues:          np.ndarray
    effective_dimension:  float
    condition_number:     float
    participation_ratio:  float
    spectral_entropy:     float
class FisherEigenvalueAnalyzer:
    def analyze(self, G: torch.Tensor) -> EigenvalueAnalysisResult:
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        d_eff   = float(ev.sum() ** 2 / (ev ** 2).sum())
        kappa   = float(ev[-1] / ev[0])
        pr      = float(ev.sum() ** 2 / ((ev ** 2).sum() * len(ev)))
        p_norm  = ev / ev.sum()
        entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
        return EigenvalueAnalysisResult(
            eigenvalues=ev,
            effective_dimension=d_eff,
            condition_number=kappa,
            participation_ratio=pr,
            spectral_entropy=entropy,
        )
    def dominant_subspace(
        self, G: torch.Tensor, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        ev, V = torch.linalg.eigh(G)
        ev = ev.cpu().numpy()
        V  = V.cpu().numpy()
        idx = np.argsort(ev)[::-1]
        return ev[idx[:n_components]], V[:, idx[:n_components]]
    def variance_explained(self, G: torch.Tensor, n_components: int) -> float:
        ev = torch.linalg.eigvalsh(G).cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        top = np.sort(ev)[::-1][:n_components]
        return float(top.sum() / ev.sum())