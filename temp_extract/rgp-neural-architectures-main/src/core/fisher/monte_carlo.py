"""
src/core/fisher/monte_carlo.py

Monte Carlo estimation of Fisher information for large networks
where exact computation is intractable.

Three estimation strategies are implemented:
  1. Hutchinson trace estimator:  Tr(F) ≈ (1/m) Σ vᵀ F v,  v ~ Rademacher(±1)
  2. Squared-gradient diagonal:   diag(F)_i ≈ E[(∂L/∂θ_i)²]
  3. Layer-metric estimator:      g^(k) ≈ (1/m) Σ_x (W_k)ᵀ W_k · scale(h^(k))

Inherits from FisherMetricBase, satisfying the abstract interface contract
and gaining access to condition_number(), effective_rank(), and
is_positive_semidefinite() as concrete utilities defined on the base class.
"""
from __future__ import annotations
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from src.core.fisher.fisher_base import FisherMetricBase


class FisherMonteCarloEstimator(FisherMetricBase):
    """
    Sampling-based Fisher information estimation.

    Implements the FisherMetricBase abstract interface (compute_layer_metric,
    compute_all_layers) plus the Hutchinson-based scalar estimators
    (estimate_trace, estimate_diagonal) and the convenience method
    estimate_layer_metric() used directly by unit tests.

    Inherited concrete methods from FisherMetricBase:
        condition_number(g)          → κ(g) = λ_max / λ_min ≥ 1
        effective_rank(g, threshold) → #{eigenvalues > threshold · λ_max}
        is_positive_semidefinite(g)  → bool (all eigenvalues ≥ tol)
    """

    def __init__(
        self,
        n_samples: int = 100,
        estimator: str = "hutchinson",
    ) -> None:
        """
        Args:
            n_samples: number of Monte Carlo samples for trace/diagonal estimation
            estimator: sampling strategy; "hutchinson" (default) or "rademacher"
                       (both use Rademacher vectors; the option exists for API
                       consistency with future Gaussian estimators)
        """
        if estimator not in ("hutchinson", "rademacher"):
            raise ValueError(f"Unknown estimator: {estimator!r}")
        self.n_samples = n_samples
        self.estimator = estimator

    # ── Abstract interface implementation (required by FisherMetricBase) ───

    def compute_layer_metric(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute the Fisher metric at a specific layer.

        Delegates to estimate_layer_metric() which uses the weight-matrix
        outer-product approximation scaled by activation variance.

        Returns:
            g: (n_in, n_in) symmetric PSD metric tensor for layer k
        """
        return self.estimate_layer_metric(model, x, layer_idx=layer_idx)

    def compute_all_layers(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Compute the Fisher metric at every linear layer in the model.

        Returns:
            List of (n_in_k, n_in_k) metric tensors, one per nn.Linear.
        """
        n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        return [self.compute_layer_metric(model, x, k) for k in range(n_linear)]

    # ── Primary estimation methods ─────────────────────────────────────────

    def estimate_layer_metric(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_idx: int = 0,
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Estimate the per-layer Fisher metric g^(k) ≈ E_x[J_k(x)ᵀ J_k(x)].

        Implementation: uses the weight matrix W_k as the layer Jacobian
        (exact for linear activations; an approximation for nonlinear ones
        near the critical initialization where tanh ≈ identity).  The result
        is scaled by the empirical activation variance at layer k to account
        for the nonlinear saturation factor.

        Args:
            model:     neural network containing nn.Linear layers
            x:         input batch of shape (batch, input_dim)
            layer_idx: 0-indexed position of the target linear layer
            n_samples: override instance-level n_samples for this call

        Returns:
            g: (n_in_k, n_in_k) symmetric PSD metric tensor

        Raises:
            IndexError: if layer_idx ≥ number of linear layers
        """
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if layer_idx >= len(linears):
            raise IndexError(
                f"layer_idx={layer_idx} out of range: "
                f"model has {len(linears)} linear layers"
            )

        W = linears[layer_idx].weight.data  # (n_out, n_in)

        # --- activation scale: forward pass to layer k ----------------------
        n_batch   = min(n_samples or self.n_samples, x.shape[0])
        idx       = torch.randperm(x.shape[0])[:n_batch]
        x_sub     = x[idx].detach()
        activations: List[torch.Tensor] = []
        call_count = [0]

        def _hook(m: nn.Module, inp, out):
            if call_count[0] == layer_idx:
                activations.append(out.detach())
            call_count[0] += 1

        handles = [m.register_forward_hook(_hook)
                   for m in model.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            model(x_sub)
        for h in handles:
            h.remove()

        # g^(k) = Wᵀ W / n_out, scaled by activation variance
        g     = W.t() @ W / max(W.shape[0], 1)
        scale = activations[0].var().item() if activations else 1.0
        return g * max(scale, 1e-8)

    def _sample_vector(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Sample a Rademacher ±1 random vector."""
        return torch.randint(0, 2, shape, device=device).float() * 2.0 - 1.0

    def estimate_trace(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Estimate Tr(F) via the Hutchinson stochastic trace estimator.

        Tr(F) ≈ (1/m) Σᵢ vᵢᵀ F vᵢ   where vᵢ ~ Rademacher(±1)

        This is an unbiased estimator with variance O(1/m), suitable for
        networks where the full O(p²) Fisher matrix is intractable.
        """
        device   = inputs.device
        params   = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        estimates = []
        for _ in range(self.n_samples):
            v        = self._sample_vector(torch.Size([n_params]), device)
            model.zero_grad()
            output   = model(inputs)
            loss     = loss_fn(output, targets)
            grads    = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            hv       = torch.autograd.grad(
                (grad_vec * v.detach()).sum(), params, retain_graph=False,
            )
            hv_vec = torch.cat([h.view(-1) for h in hv])
            estimates.append(float((v * hv_vec).sum()))

        return float(sum(estimates) / len(estimates))

    def estimate_diagonal(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the diagonal of the Fisher matrix via squared gradients.

        diag(F)_i ≈ E[(∂L/∂θ_i)²]

        Returns a non-negative vector of length equal to the total number of
        trainable parameters, suitable for use as a diagonal preconditioner.
        """
        params     = [p for p in model.parameters() if p.requires_grad]
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
                diag_accum = diag_accum + g_vec.detach() ** 2

        return diag_accum / self.n_samples
