from __future__ import annotations
import abc
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
class RGNetTemplate(nn.Module, abc.ABC):
    INPUT_DIM:   int = -1   
    HIDDEN_DIM:  int = -1
    OUTPUT_DIM:  int = -1
    DEPTH:       int = -1
    SIGMA_W:     float = 1.0   
    SIGMA_B:     float = 0.05  
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
        self._activation_cache: List[torch.Tensor] = []
        self._hook_handles: list = []
    @abc.abstractmethod
    def build_layers(self) -> nn.ModuleList:
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        activations: List[torch.Tensor] = []
        def _make_hook(layer_idx: int):
            def hook(module, inp, out):
                activations.append(out.detach())
            return hook
        handles = []
        for idx, layer in enumerate(self.layers):  
            handles.append(layer.register_forward_hook(_make_hook(idx)))
        logits = self.forward(x)
        for h in handles:
            h.remove()
        return logits, activations
    def get_layer_jacobians(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        from src.core.jacobian import AutogradJacobian
        _, acts = self.forward_with_activations(x)
        jac_computer = AutogradJacobian()
        jacobians = []
        prev = x
        for act in acts:
            jacobians.append(jac_computer.compute(prev, act))
            prev = act
        return jacobians
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def parameter_summary(self) -> Dict[str, int]:
        return {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in self.named_modules()
            if list(module.parameters(recurse=False))
        }
    def apply_critical_init(self, sigma_w: float = 1.0, sigma_b: float = 0.05) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                nn.init.normal_(module.weight, std=sigma_w / math.sqrt(fan_in))
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=sigma_b)
    @property
    def depth(self) -> int:
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