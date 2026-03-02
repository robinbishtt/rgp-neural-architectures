"""tests/stability/test_no_vanishing_gradients.py"""
import pytest
import torch
import torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


def test_critical_init_no_vanishing():
    """Critical initialization should prevent gradient vanishing."""
    depth, width = 30, 64
    layers = nn.ModuleList([StandardRGOperator(width, width) for _ in range(depth)])
    x = torch.randn(1, width, requires_grad=True)
    h = x
    for layer in layers:
        h = layer(h)
    loss = h.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    assert grad_norm > 1e-6, f"Gradient vanished: norm={grad_norm:.2e}"
 