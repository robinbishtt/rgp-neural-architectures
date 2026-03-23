"""tests/unit/test_rg_operator_shapes.py"""
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  # noqa: F811
import torch


class TestRGOperatorShapes:
    def test_standard_shape(self):
        from src.rg_flow.operators.operators import StandardRGOperator
        op = StandardRGOperator(input_dim=16, out_dim=16)
        y = op(torch.randn(4, 16))
        assert y.shape == (4, 16)

    def test_attention_shape(self):
        from src.rg_flow.operators.attention_rg_operator import AttentionRGOperator
        op = AttentionRGOperator(d_model=16, n_heads=4)
        y = op(torch.randn(4, 16))
        assert y.shape == (4, 16)

    def test_wavelet_shape(self):
        from src.rg_flow.operators.wavelet_rg_operator import WaveletRGOperator
        op = WaveletRGOperator(input_dim=16, out_dim=16)
        y = op(torch.randn(4, 16))
        assert y.shape == (4, 16)

    def test_learned_shape(self):
        from src.rg_flow.operators.learned_rg_operator import LearnedRGOperator
        op = LearnedRGOperator(input_dim=16, out_dim=16)
        y = op(torch.randn(4, 16))
        assert y.shape == (4, 16)

    def test_operators_differentiable(self):
        from src.rg_flow.operators.operators import StandardRGOperator
        op = StandardRGOperator(input_dim=8, out_dim=8)
        x = torch.randn(2, 8, requires_grad=True)
        op(x).sum().backward()
        assert x.grad is not None
 