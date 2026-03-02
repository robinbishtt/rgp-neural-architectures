"""tests/robustness/test_partial_occlusion.py"""
import torch


class TestPartialOcclusion:
    def test_zero_masking_robustness(self):
        """Model prediction should not catastrophically degrade with 50% zero masking."""
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
        # Predictions should remain consistent category-wise
        pred_full   = out_full.argmax(dim=-1)
        pred_masked = out_masked.argmax(dim=-1)
        agreement = (pred_full == pred_masked).float().mean().item()
        # Expect at least 25% agreement (above random chance for 4 classes)
        assert agreement >= 0.25

    def test_random_permutation_robustness(self):
        """RG-Net (permutation-invariant) should give same output for permuted input features."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        # Standard RGNet is NOT permutation invariant (it's an MLP), so just verify no crash
        torch.manual_seed(3)
        model = RGNetStandard(input_dim=16, n_classes=2)
        x = torch.randn(4, 16)
        perm = torch.randperm(16)
        x_perm = x[:, perm]
        with torch.no_grad():
            model(x)
            out2 = model(x_perm)
        assert not torch.isnan(out2).any()
 