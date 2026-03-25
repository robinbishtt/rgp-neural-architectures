import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
class TestOODGeneralization:
    def test_model_produces_finite_on_ood_input(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(0)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x_ood = torch.randn(16, 32) * 10.0 + 5.0
        with torch.no_grad():
            out = model(x_ood)
        assert not torch.isnan(out).any(), "NaN on OOD input"
        assert not torch.isinf(out).any(), "Inf on OOD input"
    def test_softmax_outputs_valid_probabilities_on_ood(self):
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(1)
        model = RGNetStandard(input_dim=32, n_classes=4)
        x_ood = torch.randn(8, 32) * 20.0
        with torch.no_grad():
            logits = model(x_ood)
            probs  = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()