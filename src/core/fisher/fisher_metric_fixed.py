from __future__ import annotations
import math
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False
class GradientFisherEstimator:
    def __init__(
        self,
        clip_min_eigenvalue: float = 1e-10,
        warn_gamma_threshold: float = 0.1,
    ) -> None:
        self.clip_min_ev = clip_min_eigenvalue
        self.warn_gamma  = warn_gamma_threshold
    def compute_all_layers(
        self,
        model: "nn.Module",
        inputs: "torch.Tensor",
        targets: "torch.Tensor",
        criterion: Optional["nn.Module"] = None,
        n_mc_samples: int = 1,
    ) -> List[np.ndarray]:
        if not _TORCH:
            raise ImportError("torch required")
        import torch.nn as nn
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        grads_per_layer: Dict[int, List["torch.Tensor"]] = {}
        handles = []
        layer_idx = [0]
        def _make_backward_hook(k: int):
            def hook(module, grad_input, grad_output):
                g = grad_output[0].detach()           
                if k not in grads_per_layer:
                    grads_per_layer[k] = []
                grads_per_layer[k].append(g)
            return hook
        current_k = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(
                    module.register_full_backward_hook(_make_backward_hook(current_k))
                )
                current_k += 1
        model.eval()
        inputs  = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)
        model.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        for h in handles:
            h.remove()
        eigenvalue_lists = []
        B = inputs.shape[0]
        for k in sorted(grads_per_layer.keys()):
            grad_list = grads_per_layer[k]
            if not grad_list:
                eigenvalue_lists.append(np.array([]))
                continue
            g = grad_list[0]  
            n_k = g.shape[1]
            gamma = n_k / max(B, 1)
            if gamma > self.warn_gamma:
                warnings.warn(
                    f"Layer {k}: γ = n/B = {n_k}/{B} = {gamma:.3f} > {self.warn_gamma}. "
                    f"Eigenvalue collapse risk. Need B ≥ {100*n_k} for 1% accuracy. "
                    f"Current: {100*gamma:.1f}% eigenvalue noise per Tracy-Widom.",
                    stacklevel=2,
                )
            g_np = g.cpu().float().numpy()
            F    = (g_np.T @ g_np) / max(B, 1)  
            ev = np.linalg.eigvalsh(F)
            ev = np.clip(ev, self.clip_min_ev, None)
            eigenvalue_lists.append(np.sort(ev))
        return eigenvalue_lists
    def xi_from_eigenvalues(self, eigenvalues: np.ndarray) -> float:
        pos = eigenvalues[eigenvalues > self.clip_min_ev]
        if len(pos) == 0:
            return float("nan")
        inv_mean = np.mean(1.0 / pos)
        return float(1.0 / np.sqrt(inv_mean + 1e-12))
    def xi_profile(self, eigenvalue_lists: List[np.ndarray]) -> np.ndarray:
        return np.array([self.xi_from_eigenvalues(ev) for ev in eigenvalue_lists])
class JacobianPullbackFisher:
    def compute_layer_jacobian(
        self,
        linear: "nn.Linear",
        pre_activation: "torch.Tensor",
        activation_fn: str = "tanh",
    ) -> "torch.Tensor":
        import torch
        W = linear.weight.data  
        if activation_fn == "tanh":
            dphi = 1.0 - torch.tanh(pre_activation) ** 2
        elif activation_fn == "relu":
            dphi = (pre_activation > 0).float()
        elif activation_fn == "gelu":
            cdf  = 0.5 * (1.0 + torch.erf(pre_activation / math.sqrt(2)))
            pdf  = torch.exp(-0.5 * pre_activation ** 2) / math.sqrt(2 * math.pi)
            dphi = cdf + pre_activation * pdf
        else:
            raise ValueError(f"Unknown activation: {activation_fn!r}")
        J_k = dphi.unsqueeze(1) * W   
        return J_k
    def pullback(
        self,
        G_prev: "torch.Tensor",
        J_k: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute G^(ℓ) = (J^(ℓ))ᵀ G^(ℓ-1) J^(ℓ) with PSD enforcement.

        Args:
            G_prev: Metric on the output space, shape (d_out, d_out).
            J_k:    Layer Jacobian J^(ℓ) = Diag(φ'(z)) W^(ℓ),
                    shape (d_out, d_in).

        Returns:
            Pulled-back metric of shape (d_in, d_in) with eigenvalues
            clamped to ≥ 1e-12 for numerical stability.
        """
        import torch
        G_k = J_k.T @ G_prev @ J_k   
        ev, V = torch.linalg.eigh(G_k)
        ev    = torch.clamp(ev, min=1e-12)
        return (V * ev) @ V.T
    def compute_metric_profile(
        self,
        model: "nn.Module",
        sample_input: "torch.Tensor",
        activation_fn: str = "tanh",
    ) -> Tuple[List[float], List["torch.Tensor"]]:
        import torch
        model.eval()
        x = sample_input.unsqueeze(0) if sample_input.dim() == 1 else sample_input[:1]
        x = x.to(next(model.parameters()).device)
        pre_acts = []
        hooks = []
        def _hook_pre(module, inp, out):
            with torch.no_grad():
                pre = module.linear.weight @ inp[0][0] + module.linear.bias
                pre_acts.append(pre.detach())
        for module in model.modules():
            if hasattr(module, "linear") and hasattr(module, "act"):
                hooks.append(module.register_forward_hook(_hook_pre))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        if not pre_acts:
            return [], []
        linears = [m for m in model.modules()
                   if hasattr(m, "linear") and hasattr(m, "act")]
        n0 = linears[0].linear.in_features
        G  = torch.eye(n0, device=x.device)
        eta_profile = [float(torch.linalg.eigvalsh(G).max())]
        G_matrices  = [G.clone()]
        for i, (lin_module, pre_act) in enumerate(zip(linears, pre_acts)):
            J_k = self.compute_layer_jacobian(lin_module.linear, pre_act, activation_fn)
            G   = self.pullback(G, J_k)
            eta_k = float(torch.linalg.eigvalsh(G).max())
            eta_profile.append(eta_k)
            G_matrices.append(G.clone())
        return eta_profile, G_matrices
class FisherMetric:
    def __init__(
        self,
        clip_eigenvalues: bool = True,
        min_eigenvalue: float = 1e-10,
    ) -> None:
        self.clip_eigenvalues = clip_eigenvalues
        self.min_eigenvalue   = min_eigenvalue
        self._grad_estimator  = GradientFisherEstimator(clip_min_eigenvalue=min_eigenvalue)
        self._jacobian_fisher = JacobianPullbackFisher()
    def pullback(
        self,
        G_prev: "torch.Tensor",
        J_k: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch
        G_k = J_k.T @ G_prev @ J_k
        if self.clip_eigenvalues:
            ev, V = torch.linalg.eigh(G_k)
            ev    = torch.clamp(ev, min=self.min_eigenvalue)
            G_k   = (V * ev) @ V.T
        return G_k
    def compute_from_model(
        self,
        model: "nn.Module",
        inputs: "torch.Tensor",
        targets: "torch.Tensor",
        criterion: Optional["nn.Module"] = None,
    ) -> List["torch.Tensor"]:
        import torch.nn as nn
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        ev_lists = self._grad_estimator.compute_all_layers(
            model, inputs, targets, criterion
        )
        import torch
        return [
            torch.from_numpy(ev.astype(np.float32)) if len(ev) else torch.zeros(1)
            for ev in ev_lists
        ]
    def compute_xi_profile(
        self,
        model: "nn.Module",
        inputs: "torch.Tensor",
        targets: "torch.Tensor",
        criterion: Optional["nn.Module"] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import torch.nn as nn
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        ev_lists   = self._grad_estimator.compute_all_layers(
            model, inputs, targets, criterion
        )
        xi_profile  = self._grad_estimator.xi_profile(ev_lists)
        eta_profile = np.array([
            float(ev[-1]) if len(ev) else float("nan")
            for ev in ev_lists
        ])
        return xi_profile, eta_profile