"""
src/core/fisher_metric.py

Fisher Information Geometry — layer-wise metric computation.

G^(k) = J_k G^(k-1) J_kᵀ  (pushforward)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FisherMetric:
    """
    Computes the Fisher metric G^(k) via Jacobian pushforward.

    G^(k) = J_k G^(k-1) J_kᵀ

    where J_k = ∂h^(k)/∂h^(k-1) is the layer Jacobian.
    """

    def __init__(
        self,
        clip_eigenvalues: bool = True,
        min_eigenvalue:   float = 1e-10,
    ) -> None:
        self.clip_eigenvalues = clip_eigenvalues
        self.min_eigenvalue   = min_eigenvalue

    def pushforward(
        self,
        G_prev: torch.Tensor,
        J_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute G^(k) = J_k G^(k-1) J_kᵀ.

        Parameters
        ----------
        G_prev : (N_{k-1}, N_{k-1})  metric at layer k-1
        J_k    : (N_k, N_{k-1})      layer Jacobian

        Returns
        -------
        G_k    : (N_k, N_k)           metric at layer k
        """
        G_k = J_k @ G_prev @ J_k.T

        if self.clip_eigenvalues:
            G_k = self._clip(G_k)

        return G_k

    def _clip(self, G: torch.Tensor) -> torch.Tensor:
        """Clip negative eigenvalues for PSD enforcement."""
        ev, V = torch.linalg.eigh(G)
        ev    = torch.clamp(ev, min=self.min_eigenvalue)
        return V @ torch.diag(ev) @ V.T

    def compute_from_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Compute layer-wise Fisher metrics via autograd Jacobian hooks.

        Returns list of G^(k) tensors for requested layers.
        """
        metrics: List[torch.Tensor] = []
        activations: List[torch.Tensor] = []
        hooks = []

        def _hook(module, inp, out):
            activations.append(out.detach())

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(_hook))

        with torch.enable_grad():
            x = x.requires_grad_(True)
            y = model(x)

        for h in hooks:
            h.remove()

        # Compute identity G^(0) and propagate
        if not activations:
            return metrics

        n0 = activations[0].shape[-1]
        G  = torch.eye(n0, dtype=activations[0].dtype, device=activations[0].device)

        for i, act in enumerate(activations):
            if layer_indices is None or i in layer_indices:
                n = act.shape[-1]
                # Approximate Jacobian via output covariance
                a  = act.view(-1, n)
                J  = a.T @ a / a.shape[0]
                G  = self.pushforward(G[:n, :n] if G.shape[0] != n else G, J)
                metrics.append(G.clone())

        return metrics


class FisherEigenvalueAnalyzer:
    """Spectral decomposition and eigenvalue density estimation of Fisher matrices."""

    def analyze(
        self,
        G: torch.Tensor,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Returns (eigenvalues, effective_dimension, condition_number).
        """
        ev    = torch.linalg.eigvalsh(G).cpu().numpy()
        ev    = np.clip(ev, 1e-12, None)
        d_eff = float((ev.sum() ** 2) / (ev ** 2).sum())
        kappa = float(ev[-1] / ev[0])
        return ev, d_eff, kappa
 