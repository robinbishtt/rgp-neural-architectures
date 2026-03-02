"""
src/architectures/rg_net/rg_net_template.py

RGNetTemplate: abstract base class defining the RG-Net interface.

All RG-Net variants (Shallow, Standard, Deep, UltraDeep, VariableWidth,
MultiScale) inherit from this template.  It enforces:
  * A common forward/forward_with_activations API
  * Layer-wise Jacobian access for Fisher metric computation
  * Correlation-length hooks for H1 validation
  * Critical initialisation bookkeeping

Design rationale:
    The template decouples *architectural topology* from *mathematical
    infrastructure*.  Tier 1 modules (Fisher, Jacobian, Lyapunov) import
    from this module, not from specific variants.  This guarantees that
    adding a new architecture variant does not break any existing test.
"""
from __future__ import annotations

import abc
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RGNetTemplate(nn.Module, abc.ABC):
    """
    Abstract base class for all RG-Net architectures.

    Subclasses must implement:
        * build_layers() -> nn.ModuleList
        * forward(x) -> Tensor

    Optional overrides:
        * forward_with_activations(x) -> Tuple[Tensor, List[Tensor]]
        * get_jacobians(x) -> List[Tensor]
    """

    # ------------------------------------------------------------------
    # Canonical hyperparameter registry
    # ------------------------------------------------------------------
    INPUT_DIM:   int = -1   # set by subclass __init__
    HIDDEN_DIM:  int = -1
    OUTPUT_DIM:  int = -1
    DEPTH:       int = -1
    SIGMA_W:     float = 1.0   # weight init std multiplier
    SIGMA_B:     float = 0.05  # bias init std

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        depth:      int,
        activation: str   = "tanh",
        sigma_w:    float = 1.0,
        sigma_b:    float = 0.05,
    ) -> None:
        super().__init__()
        self.INPUT_DIM  = input_dim
        self.HIDDEN_DIM = hidden_dim
        self.OUTPUT_DIM = output_dim
        self.DEPTH      = depth
        self.SIGMA_W    = sigma_w
        self.SIGMA_B    = sigma_b
        self.activation_name = activation

        # Registered activation hook outputs for Jacobian / Fisher access
        self._activation_cache: List[torch.Tensor] = []
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_layers(self) -> nn.ModuleList:
        """
        Construct and return the ordered ModuleList of RGLayer instances.
        Called once from __init__ of each concrete subclass.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning logits."""

    # ------------------------------------------------------------------
    # Optional overrides — default implementations provided
    # ------------------------------------------------------------------

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that also returns per-layer activations.

        Returns
        -------
        logits : Tensor  (B, output_dim)
        activations : List[Tensor]  one entry per hidden layer (B, hidden_dim)
        """
        activations: List[torch.Tensor] = []

        def _make_hook(layer_idx: int):
            def hook(module, inp, out):
                activations.append(out.detach())
            return hook

        handles = []
        for idx, layer in enumerate(self.layers):  # type: ignore[attr-defined]
            handles.append(layer.register_forward_hook(_make_hook(idx)))

        logits = self.forward(x)
        for h in handles:
            h.remove()
        return logits, activations

    def get_layer_jacobians(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Compute per-layer Jacobians dh^(k)/dh^(k-1) via autograd.

        Returns a list of Jacobian matrices, one per hidden layer.
        For width N and batch size B this is a list of (B, N, N) tensors.

        Note: This is the *exact* Jacobian.  For large widths (N > 512)
        use the JVP or VJP variants in src/core/jacobian/ instead.
        """
        from src.core.jacobian import AutogradJacobian
        _, acts = self.forward_with_activations(x)
        jac_computer = AutogradJacobian()
        jacobians = []
        prev = x
        for act in acts:
            jacobians.append(jac_computer.compute(prev, act))
            prev = act
        return jacobians

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """Return per-module trainable parameter counts."""
        return {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in self.named_modules()
            if list(module.parameters(recurse=False))
        }

    def apply_critical_init(self, sigma_w: float = 1.0, sigma_b: float = 0.05) -> None:
        """
        Re-apply critical initialisation σ_w = 1, σ_b = 0.05 (edge-of-chaos).

        This can be called after loading a checkpoint to reset parameters to
        the critical manifold before fine-tuning.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                nn.init.normal_(module.weight, std=sigma_w / math.sqrt(fan_in))
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=sigma_b)

    @property
    def depth(self) -> int:
        """Alias for DEPTH to match manuscript terminology."""
        return self.DEPTH

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input={self.INPUT_DIM}, "
            f"hidden={self.HIDDEN_DIM}, "
            f"output={self.OUTPUT_DIM}, "
            f"depth={self.DEPTH}, "
            f"params={self.count_parameters():,})"
        )
 