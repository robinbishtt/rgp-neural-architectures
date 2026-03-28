import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
class TestMixedPrecisionStability:
    def test_fp16_forward_no_nan(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(11)
        model = RGNetStandard(input_dim=32, n_classes=4).half()
        x = torch.randn(8, 32).half()
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "NaN in FP16 forward pass"
    def test_fp32_fp16_consistency(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(12)
        model32 = RGNetStandard(input_dim=16, n_classes=2)
        model16 = RGNetStandard(input_dim=16, n_classes=2)
        model16.load_state_dict(model32.state_dict())
        model16 = model16.half()
        x = torch.randn(4, 16)
        with torch.no_grad():
            out32 = model32(x)
            out16 = model16(x.half()).float()
        # Deep tanh stacks accumulate FP16 quantization error in logit scale.
        # Keep strict class-consistency plus bounded absolute deviation.
        assert torch.allclose(out32, out16, atol=2.5e-1), "FP32 and FP16 outputs differ too much"
        assert torch.equal(out32.argmax(dim=-1), out16.argmax(dim=-1)), \
            "FP32/FP16 predicted classes should match"
