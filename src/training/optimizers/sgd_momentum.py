"""
src/training/optimizers/sgd_momentum.py

SGD with Nesterov momentum and adaptive scheduling.
"""
from __future__ import annotations
import torch


def build_sgd(params, lr: float = 0.01, momentum: float = 0.9,
              weight_decay: float = 1e-4, nesterov: bool = True):
    return torch.optim.SGD(params, lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=nesterov)
 