import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn
import numpy as np
def _make_model(width: int = 32) -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, 2),
    )
def _correlated_input(batch: int, dim: int, correlation: float, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal((batch, 1))
    eps = rng.standard_normal((batch, dim))
    x   = np.sqrt(correlation) * z + np.sqrt(max(1.0 - correlation, 0.0)) * eps
    return torch.tensor(x, dtype=torch.float32)
@pytest.fixture
def model():
    return _make_model()
class TestDistributionShift:
    def test_iid_input_output_finite(self, model):
        x = _correlated_input(32, 32, correlation=0.0)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
    def test_high_correlation_output_finite(self, model):
        x = _correlated_input(32, 32, correlation=0.9)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
    def test_fully_correlated_output_finite(self, model):
        x = _correlated_input(32, 32, correlation=1.0)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
    def test_output_shape_invariant_to_shift(self, model):
        for corr in [0.0, 0.3, 0.6, 0.9]:
            x = _correlated_input(16, 32, correlation=corr)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (16, 2), (
                f"Output shape {out.shape} != (16, 2) at correlation={corr}."
            )
    def test_output_entropy_bounded_under_shift(self, model):
        import torch.nn.functional as F
        import math
        max_entropy = math.log(2)  
        for corr in [0.0, 0.5, 0.9]:
            x = _correlated_input(32, 32, correlation=corr)
            with torch.no_grad():
                logits = model(x)
                probs  = F.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            assert entropy > 0.0, f"Zero entropy at correlation={corr} - model collapsed."
            assert entropy <= max_entropy + 1e-4, (
                f"Entropy {entropy:.4f} > max={max_entropy:.4f} at correlation={corr}."
            )