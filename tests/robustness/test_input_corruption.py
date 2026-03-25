import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
def _make_model(width: int = 32) -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, 2),
    )
@pytest.fixture
def model():
    return _make_model()
@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(16, 32)
class TestInputCorruption:
    def test_zero_masking_finite(self, model, x):
        mask = torch.bernoulli(torch.full_like(x, 0.5))
        x_corrupted = x * mask
        with torch.no_grad():
            out = model(x_corrupted)
        assert torch.isfinite(out).all(), "NaN/Inf after 50% zero-masking."
    def test_full_zero_input_finite(self, model, x):
        x_zero = torch.zeros_like(x)
        with torch.no_grad():
            out = model(x_zero)
        assert torch.isfinite(out).all(), "NaN/Inf for all-zero input."
    def test_single_feature_active(self, model, x):
        for feat_idx in range(0, x.shape[1], 8):
            x_sparse = torch.zeros_like(x)
            x_sparse[:, feat_idx] = x[:, feat_idx]
            with torch.no_grad():
                out = model(x_sparse)
            assert torch.isfinite(out).all(), (
                f"NaN/Inf with single feature {feat_idx} active."
            )
    def test_random_masking_levels(self, model, x):
        for keep_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mask = torch.bernoulli(torch.full_like(x, keep_prob))
            x_corrupted = x * mask
            with torch.no_grad():
                out = model(x_corrupted)
            assert torch.isfinite(out).all(), (
                f"NaN/Inf at keep_prob={keep_prob}."
            )
    def test_output_shape_invariant_to_corruption(self, model, x):
        x_corrupted = x * torch.bernoulli(torch.full_like(x, 0.5))
        with torch.no_grad():
            out_clean = model(x)
            out_corrupted = model(x_corrupted)
        assert out_clean.shape == out_corrupted.shape, (
        )