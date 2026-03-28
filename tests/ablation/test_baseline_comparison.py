import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
class TestBaselineComparison:
    def _get_all_models(self, input_dim=16, n_classes=4):
        from src.architectures.baselines.mlp_baseline import MLPBaseline
        from src.architectures.baselines.resnet_baseline import ResNetBaseline
        from src.architectures.baselines.transformer_baseline import TransformerBaseline
        from src.architectures.baselines.inception_baseline import InceptionBaseline
        return {
            "mlp":         MLPBaseline(input_dim=input_dim, n_classes=n_classes),
            "resnet":      ResNetBaseline(input_dim=input_dim, n_classes=n_classes),
            "transformer": TransformerBaseline(input_dim=input_dim, n_classes=n_classes,
                                               d_model=32, n_heads=4, n_layers=2),
            "inception":   InceptionBaseline(input_dim=input_dim, n_classes=n_classes),
        }
    def test_all_output_correct_shape(self):
        torch.manual_seed(5)
        models = self._get_all_models()
        x = torch.randn(8, 16)
        for name, model in models.items():
            with torch.no_grad():
                out = model(x)
            assert out.shape == (8, 4), f"{name}: wrong output shape {out.shape}"
    def test_all_produce_finite_outputs(self):
        torch.manual_seed(6)
        models = self._get_all_models()
        x = torch.randn(8, 16)
        for name, model in models.items():
            with torch.no_grad():
                out = model(x)
            assert not torch.isnan(out).any(), f"{name}: NaN in output"
            assert not torch.isinf(out).any(), f"{name}: Inf in output"