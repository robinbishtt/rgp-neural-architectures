"""
src/core/correlation/transfer_matrix.py

TransferMatrixMethod: computes correlation lengths from eigenvalue ratios
of the layer transfer matrix, providing an alternative to the Fisher
spectrum method that is exact for linear models and serves as a
cross-validation tool for nonlinear architectures.

Reference: Cardy, J. (1988). Finite-Size Scaling. North-Holland.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TransferMatrixMethod:
    """
    Transfer matrix approach to correlation length estimation.

    For a linearized layer transformation T_k = J_k (Jacobian matrix),
    the correlation length is:
        ξ_TM(k) = -1 / log(|λ₁ / λ₀|)
    where λ₀ ≥ |λ₁| are the two largest eigenvalues of Tᵀ T.

    This is equivalent to the ratio of the two largest singular values:
        ξ_TM(k) = -1 / log(σ₁(k) / σ₀(k))

    Physical interpretation: σ₀ governs the leading-order information
    propagation; σ₁ / σ₀ < 1 ensures that subleading modes decay, and
    ξ_TM quantifies the characteristic depth over which this decay occurs.
    """

    def __init__(self, top_k: int = 2) -> None:
        """
        Args:
            top_k: number of top singular values to retain for ratio computation.
                   Must be ≥ 2. Higher values allow cross-checks on the ratio
                   stability across the spectrum.
        """
        if top_k < 2:
            raise ValueError(f"top_k must be ≥ 2, got {top_k}")
        self.top_k = top_k

    def compute_from_jacobian(
        self, J: torch.Tensor
    ) -> float:
        """
        Estimate ξ_TM from a single layer Jacobian matrix.

        Args:
            J: (n_out, n_in) Jacobian tensor for one layer.

        Returns:
            Correlation length estimate ξ_TM > 0.
            Returns np.inf if σ₀ ≈ σ₁ (near-critical, slow decay).
        """
        J_np  = J.detach().cpu().numpy()
        svs   = np.linalg.svd(J_np, compute_uv=False)
        if len(svs) < 2 or svs[0] < 1e-12:
            return float("inf")
        ratio = svs[1] / (svs[0] + 1e-12)
        ratio = np.clip(ratio, 1e-12, 1.0 - 1e-12)
        return float(-1.0 / np.log(ratio))

    def compute_depth_profile(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute per-layer ξ_TM by extracting Jacobians via hooks.

        Args:
            model: network with nn.Linear layers
            x:     input tensor (single sample, no batch dimension needed)

        Returns:
            Array of correlation length estimates per layer, shape (n_layers,)
        """
        xi_values: list = []

        def _make_hook(results):
            def hook(module: nn.Module, inp, out):
                W = module.weight.data
                results.append(W.cpu())
            return hook

        handles = []
        Ws: list = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(_make_hook(Ws)))

        with torch.no_grad():
            _ = model(x)
        for h in handles:
            h.remove()

        for W in Ws:
            xi_values.append(self.compute_from_jacobian(W))

        return np.array(xi_values)

    def gap_ratio(
        self, J: torch.Tensor, k: int = 2
    ) -> float:
        """
        Compute the spectral gap ratio σₖ₋₁ / σ₀ for the first k singular values.

        Used as a stability diagnostic: larger gap ratios indicate faster
        correlation length decay and more robust information bottlenecking.

        Args:
            J: layer Jacobian
            k: index of the subleading singular value (default 2, i.e., σ₁/σ₀)

        Returns:
            Gap ratio in [0, 1]
        """
        J_np = J.detach().cpu().numpy()
        svs  = np.linalg.svd(J_np, compute_uv=False)
        if len(svs) < k or svs[0] < 1e-12:
            return 0.0
        return float(svs[k - 1] / (svs[0] + 1e-12))
 