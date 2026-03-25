import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
class TestPartialOcclusion:
    def test_zero_masking_robustness(self):
        from src.architectures.rg_net.rg_net_multiscale import RGNetMultiScale
        torch.manual_seed(2)
        model = RGNetMultiScale(input_dim=32, n_classes=4)
        model.eval()
        x = torch.randn(16, 32)
        mask = torch.bernoulli(0.5 * torch.ones(16, 32))
        x_masked = x * mask
        with torch.no_grad():
            out_full   = model(x)
            out_masked = model(x_masked)
        assert not torch.isnan(out_masked).any(), "NaN with zero masking"
        pred_full   = out_full.argmax(dim=-1)
        pred_masked = out_masked.argmax(dim=-1)
        agreement = (pred_full == pred_masked).float().mean().item()
        assert agreement >= 0.25
    def test_random_permutation_robustness(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(3)
        model = RGNetStandard(input_dim=16, n_classes=2)
        x = torch.randn(4, 16)
        perm = torch.randperm(16)
        x_perm = x[:, perm]
        with torch.no_grad():
            model(x)
            out2 = model(x_perm)
        assert not torch.isnan(out2).any()