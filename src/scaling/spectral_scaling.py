from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
@dataclass
class SpectralScalingResult:
    layer_index:      int
    singular_values:  np.ndarray
    bulk_edge:        float         
    max_sv:           float         
    edge_gap:         float         
    effective_rank:   int
    mp_ks_stat:       Optional[float]  
    mp_p_value:       Optional[float]
class SpectralScalingAnalyzer:
    def analyze_layer(
        self,
        W:     torch.Tensor,
        sigma2: float = 1.0,
        layer_index: int = 0,
    ) -> SpectralScalingResult:
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution
        W_np = W.detach().cpu().numpy()
        n, m = W_np.shape
        beta = n / m
        svs = np.linalg.svd(W_np, compute_uv=False)
        lambdas = svs ** 2 / m  
        mp = MarchenkoPasturDistribution(beta=beta, sigma2=sigma2)
        ks_stat, p_val = mp.ks_test(lambdas)
        eff_rank = int(np.sum(svs > 0.01 * svs[0]))
        return SpectralScalingResult(
            layer_index=layer_index,
            singular_values=svs,
            bulk_edge=float(mp.lam_plus),
            max_sv=float(svs[0]),
            edge_gap=float((np.max(lambdas) - mp.lam_plus) / (mp.lam_plus + 1e-12)),
            effective_rank=eff_rank,
            mp_ks_stat=ks_stat,
            mp_p_value=p_val,
        )
    def analyze_model(
        self, model: nn.Module, sigma2: float = 1.0
    ) -> List[SpectralScalingResult]:
        results = []
        k = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                results.append(self.analyze_layer(m.weight.data, sigma2, k))
                k += 1
        return results