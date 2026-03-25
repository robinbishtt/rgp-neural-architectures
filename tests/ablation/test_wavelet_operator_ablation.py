import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
class TestWaveletOperatorAblation:
    def test_with_and_without_detail(self):
        from src.rg_flow.operators.wavelet_rg_operator import WaveletRGOperator
        torch.manual_seed(1)
        op_with    = WaveletRGOperator(in_dim=16, out_dim=16, use_detail=True)
        op_without = WaveletRGOperator(in_dim=16, out_dim=16, use_detail=False)
        x = torch.randn(4, 16)
        with torch.no_grad():
            y_with    = op_with(x)
            y_without = op_without(x)
        assert not torch.isnan(y_with).any()
        assert not torch.isnan(y_without).any()
    def test_gradient_through_detail_mixing(self):
        from src.rg_flow.operators.wavelet_rg_operator import WaveletRGOperator
        op = WaveletRGOperator(in_dim=8, out_dim=8, use_detail=True)
        x  = torch.randn(2, 8, requires_grad=True)
        op(x).sum().backward()
        assert x.grad is not None