"""
src/training/optimizers/layer_wise.py

Layer-wise learning rate assignment for fine-grained depth control.
"""
from __future__ import annotations
import torch.nn as nn
import torch


def build_layerwise_adam(
    model: nn.Module,
    base_lr: float = 1e-3,
    lr_decay: float = 0.9,
    weight_decay: float = 1e-4,
) -> torch.optim.Adam:
    """
    Assign exponentially decaying learning rates per layer.
    Early layers (close to input) receive smaller LR; later layers larger.
    """
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    n = len(named_params)
    param_groups = []
    for i, (name, p) in enumerate(named_params):
        layer_lr = base_lr * (lr_decay ** (n - i - 1))
        param_groups.append({"params": [p], "lr": layer_lr, "name": name})
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)
 