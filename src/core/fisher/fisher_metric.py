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
        G_k = J_k.T @ G_prev @ J_k
        if self.clip_eigenvalues:
            G_k = self._clip(G_k)
        return G_k
    pushforward = pullback  
    def _clip(self, G: torch.Tensor) -> torch.Tensor:
        ev, V = torch.linalg.eigh(G)
        ev    = torch.clamp(ev, min=self.min_eigenvalue)
        return V @ torch.diag(ev) @ V.T
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
    def analyze(
        self,
        G: torch.Tensor,
    ) -> Tuple[np.ndarray, float, float]:
        ev    = torch.linalg.eigvalsh(G).cpu().numpy()
        ev    = np.clip(ev, 1e-12, None)
        d_eff = float((ev.sum() ** 2) / (ev ** 2).sum())
        kappa = float(ev[-1] / ev[0])
        return ev, d_eff, kappa