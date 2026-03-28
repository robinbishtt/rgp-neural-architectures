import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
from src.rg_flow.operators.operators import StandardRGOperator, ResidualRGOperator
def test_residual_vs_standard_gradient_flow():
    depth  = 15
    width  = 32
    std_layers = [StandardRGOperator(width, width) for _ in range(depth)]
    res_layers = [ResidualRGOperator(width, width) for _ in range(depth)]
    for group in [std_layers, res_layers]:
        # Use a fresh leaf tensor (detach from any computation graph)
        xc = torch.randn(1, width, requires_grad=True)
        h  = xc
        for l in group:
            h = l(h)
        h.sum().backward()
        assert xc.grad is not None