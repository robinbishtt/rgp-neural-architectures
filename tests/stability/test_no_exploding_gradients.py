"""tests/stability/test_no_exploding_gradients.py"""
import torch
import torch.nn as nn
from src.rg_flow.operators.operators import StandardRGOperator


def test_critical_init_no_exploding():
    depth, width = 30, 64
    layers = nn.ModuleList([StandardRGOperator(width, width) for _ in range(depth)])
    x = torch.randn(1, width, requires_grad=True)
    h = x
    for layer in layers:
        h = layer(h)
    loss = h.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    assert grad_norm < 1e4, f"Gradient exploded: norm={grad_norm:.2e}"
 