"""
src/scaling/spectral_scaling.py

SpectralScalingAnalyzer: analyzes how Jacobian spectral statistics
(Marchenko-Pastur bulk, Wigner edge, level spacing) scale with network
depth L and width N.

Used in Extended Data figures (run_extended_figure2.py) and in the
spectral tests (tests/spectral/) to verify RMT predictions.
"""
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
    bulk_edge:        float         # Marchenko-Pastur λ₊
    max_sv:           float         # empirical maximum singular value
    edge_gap:         float         # (max_sv - bulk_edge) / bulk_edge
    effective_rank:   int
    mp_ks_stat:       Optional[float]  # KS statistic vs MP
    mp_p_value:       Optional[float]


class SpectralScalingAnalyzer:
    """
    Extracts and analyzes Jacobian spectral statistics per network layer.

    The central quantity is the empirical spectral density (ESD) of the
    layer Jacobian's singular value distribution. Under the RGP theory,
    layers near the critical point should have an ESD matching the
    Marchenko-Pastur distribution, while deviations indicate departures
    from criticality.
    """

    def analyze_layer(
        self,
        W:     torch.Tensor,
        sigma2: float = 1.0,
        layer_index: int = 0,
    ) -> SpectralScalingResult:
        """
        Analyze the singular value spectrum of a single layer weight matrix.

        Args:
            W:           (n_out, n_in) weight matrix tensor
            sigma2:      entry variance (for MP parameter estimation)
            layer_index: for bookkeeping

        Returns:
            SpectralScalingResult with bulk statistics.
        """
        from src.core.spectral.marchenko_pastur import MarchenkoPasturDistribution

        W_np = W.detach().cpu().numpy()
        n, m = W_np.shape
        beta = n / m

        svs = np.linalg.svd(W_np, compute_uv=False)
        lambdas = svs ** 2 / m  # convert to eigenvalues of (1/m) WᵀW

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
        """
        Analyze spectral statistics for all linear layers in a model.

        Returns:
            List of SpectralScalingResult, one per nn.Linear layer.
        """
        results = []
        k = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                results.append(self.analyze_layer(m.weight.data, sigma2, k))
                k += 1
        return results
 