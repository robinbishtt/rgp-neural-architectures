"""
src/core/fisher/fisher_base.py

Abstract base class for all Fisher information metric estimators.
Provides the common interface contract that FisherMetric, FisherMonteCarloEstimator,
and FisherAnalyticCalculator must satisfy.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FisherMetricBase(ABC):
    """
    Abstract base class defining the interface for Fisher metric computation.

    All Fisher estimators in src/core/fisher/ implement this interface,
    enabling polymorphic use in the training loop and hypothesis validation
    experiments (H1 scale correspondence, H2 depth scaling).

    The Fisher metric g^(k) at layer k satisfies the RG transformation law:
        g^(k) = Jₖᵀ g^(k-1) Jₖ
    where Jₖ = ∂h^(k)/∂h^(k-1) is the layer Jacobian.
    """

    @abstractmethod
    def compute_layer_metric(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute the Fisher information metric at a specific layer.

        Args:
            model:     neural network
            x:         input batch tensor
            layer_idx: index of the layer (0-indexed from input)

        Returns:
            g: (N, N) symmetric positive semi-definite metric tensor
        """

    @abstractmethod
    def compute_all_layers(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> list:
        """
        Compute Fisher metric at all layers.

        Args:
            model: neural network
            x:     input batch tensor

        Returns:
            List of (N_k, N_k) metric tensors, one per layer.
        """

    def condition_number(self, g: torch.Tensor) -> float:
        """
        Compute the condition number of the metric tensor.

        κ(g) = λ_max / λ_min. High condition numbers indicate
        near-singular metrics and potential optimization difficulties.

        Args:
            g: (N, N) metric tensor

        Returns:
            condition number κ ≥ 1
        """
        evs = torch.linalg.eigvalsh(g)
        evs = evs[evs > 0]
        if len(evs) == 0:
            return float("inf")
        return float(evs[-1] / (evs[0] + 1e-12))

    def effective_rank(self, g: torch.Tensor, threshold: float = 0.01) -> int:
        """
        Compute the effective rank of the metric tensor via thresholded eigenvalues.

        Effective rank = number of eigenvalues > threshold * λ_max.

        Args:
            g:         (N, N) metric tensor
            threshold: fraction of maximum eigenvalue for rank cutoff

        Returns:
            effective rank (integer)
        """
        evs = torch.linalg.eigvalsh(g)
        evs = evs[evs > 0]
        if len(evs) == 0:
            return 0
        cutoff = threshold * evs[-1]
        return int((evs > cutoff).sum().item())

    def is_positive_semidefinite(
        self, g: torch.Tensor, tol: float = -1e-6
    ) -> bool:
        """
        Check whether the metric tensor is positive semi-definite.

        Args:
            g:   (N, N) tensor
            tol: tolerance for negative eigenvalues (default -1e-6)

        Returns:
            True if all eigenvalues ≥ tol
        """
        evs = torch.linalg.eigvalsh(g)
        return bool((evs >= tol).all().item())
 