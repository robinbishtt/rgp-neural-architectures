"""tests/ablation/test_transformer_baseline_ablation.py"""
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch


class TestTransformerBaselineAblation:
    def test_transformer_output_shape(self):
        from src.architectures.baselines.transformer_baseline import TransformerBaseline
        model = TransformerBaseline(input_dim=16, n_classes=4, d_model=32, n_heads=4, n_layers=2)
        out = model(torch.randn(8, 16))
        assert out.shape == (8, 4)

    def test_transformer_training_step(self):
        import torch.nn as nn
        import torch.optim as optim
        from src.architectures.baselines.transformer_baseline import TransformerBaseline
        torch.manual_seed(3)
        model = TransformerBaseline(input_dim=16, n_classes=4, d_model=32, n_heads=4, n_layers=2)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        x, y = torch.randn(8, 16), torch.randint(0, 4, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert not torch.isnan(loss)
 