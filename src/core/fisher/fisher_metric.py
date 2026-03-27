from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False
    torch = None
    nn    = None
class FisherMetric:
    """Layer-wise Fisher Information Metric via the Riemannian pullback.

    Given the Jacobian J^(ℓ) ∈ ℝ^{d_ℓ × d_{ℓ-1}} of layer ℓ and the metric
    G^(ℓ-1) on the previous layer's representation space, the pullback metric
    on the input space is::

        G^(ℓ) = (J^(ℓ))ᵀ G^(ℓ-1) J^(ℓ)   ∈ ℝ^{d_{ℓ-1} × d_{ℓ-1}}

    Initialise with G^(0) = I (flat background metric) and iterate through all
    layers.  The spectral norm ‖G^(ℓ)‖₂ = η^(ℓ) satisfies the contraction
    bound η^(ℓ) ≤ χ₁ · η^(ℓ-1), where χ₁ = σ_w² · E[φ'(z)²] ≤ 1 at the
    critical initialisation.

    Args:
        clip_eigenvalues: If ``True``, clamp negative eigenvalues of G^(ℓ) to
                          ``min_eigenvalue`` after each pullback to maintain
                          positive semi-definiteness in the presence of
                          floating-point round-off.
        min_eigenvalue:   Floor value used when ``clip_eigenvalues`` is ``True``.
    """

    def __init__(
        self,
        clip_eigenvalues: bool = True,
        min_eigenvalue:   float = 1e-10,
    ) -> None:
        self.clip_eigenvalues = clip_eigenvalues
        self.min_eigenvalue   = min_eigenvalue
    def pullback(
        self,
        G_prev: torch.Tensor,
        J_k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Riemannian pullback G^(ℓ) = (J^(ℓ))ᵀ G^(ℓ-1) J^(ℓ).

        Args:
            G_prev: Metric tensor on the output space, shape (d_out, d_out).
            J_k:    Jacobian of layer ℓ, shape (d_out, d_in).

        Returns:
            Pulled-back metric G^(ℓ) of shape (d_in, d_in), optionally with
            eigenvalues clipped to ``min_eigenvalue``.
        """
        G_k = J_k.T @ G_prev @ J_k
        if self.clip_eigenvalues:
            G_k = self._clip(G_k)
        return G_k
    def _clip(self, G: torch.Tensor) -> torch.Tensor:
        """Project G onto the PSD cone by clamping eigenvalues.

        Uses the spectral decomposition G = V Λ Vᵀ and replaces each
        λ_i < ``min_eigenvalue`` with ``min_eigenvalue``.  The reconstruction
        uses ``(V * ev) @ V.T`` which avoids forming a dense diagonal matrix
        via ``torch.diag`` and is therefore O(n²) rather than O(n²·n) in
        memory.
        """
        ev, V = torch.linalg.eigh(G)
        ev    = torch.clamp(ev, min=self.min_eigenvalue)
        return (V * ev) @ V.T
    def compute_from_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
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
            model(x)
        for h in hooks:
            h.remove()
        if not activations:
            return metrics
        n0 = activations[0].shape[-1]
        G  = torch.eye(n0, dtype=activations[0].dtype, device=activations[0].device)
        for i, act in enumerate(activations):
            if layer_indices is None or i in layer_indices:
                n = act.shape[-1]
                a  = act.view(-1, n)
                sigma2 = (a ** 2).mean().clamp(min=1e-10)
                J_approx = torch.eye(min(n, G.shape[0]),
                                     dtype=act.dtype, device=act.device) * sigma2.sqrt()
                G_in  = G[:J_approx.shape[0], :J_approx.shape[0]]
                G  = self.pullback(G_in, J_approx)
                metrics.append(G.clone())
        return metrics
class FisherEigenvalueAnalyzer:
    """Analyse the eigenvalue spectrum of a Fisher metric tensor G.

    Computes standard spectral summaries including the effective dimension
    d_eff = (Tr G)² / Tr(G²) (participation ratio) and the condition number
    κ(G) = λ_max / λ_min.
    """

    def analyze(
        self,
        G: torch.Tensor,
    ) -> Tuple[np.ndarray, float, float]:
        """Return eigenvalues, effective dimension, and condition number.

        Args:
            G: Symmetric positive-semi-definite metric tensor, shape (n, n).

        Returns:
            Tuple ``(ev, d_eff, kappa)`` where:

            - ``ev``     – sorted eigenvalues as a NumPy array (ascending).
            - ``d_eff``  – participation ratio (Tr G)² / Tr(G²).
            - ``kappa``  – condition number λ_max / λ_min (clamped away from 0).
        """
        ev    = torch.linalg.eigvalsh(G).cpu().numpy()
        ev    = np.clip(ev, 1e-12, None)
        d_eff = float((ev.sum() ** 2) / (ev ** 2).sum())
        kappa = float(ev[-1] / ev[0])
        return ev, d_eff, kappa