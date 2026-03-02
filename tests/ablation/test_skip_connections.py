"""tests/ablation/test_skip_connections.py"""
import torch
from src.rg_flow.operators.operators import StandardRGOperator, ResidualRGOperator


def test_residual_vs_standard_gradient_flow():
    depth  = 15
    width  = 32
    std_layers = [StandardRGOperator(width, width) for _ in range(depth)]
    res_layers = [ResidualRGOperator(width, width) for _ in range(depth)]
    x = torch.randn(1, width, requires_grad=True)
    for group in [std_layers, res_layers]:
        xc = x.clone().requires_grad_(True)
        h  = xc
        for l in group:
            h = l(h)
        h.sum().backward()
        assert xc.grad is not None
 