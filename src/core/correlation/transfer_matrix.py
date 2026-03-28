from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

EPSILON_STABILITY = 1e-12
RATIO_MIN_CLIP = 1e-12
RATIO_MAX_CLIP = 1.0 - 1e-12

class TransferMatrixMethod:
    def __init__(self, top_k: int = 2) -> None:
        if top_k < 2:
            raise ValueError(f"top_k must be ≥ 2, got {top_k}")
        self.top_k = top_k
    def compute_from_jacobian(
        self, J: torch.Tensor
    ) -> float:
        J_np  = J.detach().cpu().float().numpy()
        svs   = np.linalg.svd(J_np, compute_uv=False)
        if len(svs) < 2 or svs[0] < EPSILON_STABILITY:
            return float("inf")
        ratio = float(svs[1]) / (float(svs[0]) + EPSILON_STABILITY)
        ratio = np.clip(ratio, RATIO_MIN_CLIP, RATIO_MAX_CLIP)
        return float(-1.0 / np.log(ratio))
    def compute_depth_profile(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> np.ndarray:
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
        J_np = J.detach().cpu().float().numpy()
        svs  = np.linalg.svd(J_np, compute_uv=False)
        if len(svs) < k or svs[0] < EPSILON_STABILITY:
            return 0.0
        return float(svs[k - 1] / (svs[0] + EPSILON_STABILITY))
