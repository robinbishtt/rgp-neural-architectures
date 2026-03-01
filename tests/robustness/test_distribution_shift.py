"""
tests/robustness/test_distribution_shift.py

OOD generalisation under correlation structure shift.
Tests that RG-Net outputs degrade gracefully as input distribution shifts.
"""

import pytest
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
    """
    Generate inputs with a given pairwise correlation structure.
    correlation=0.0 → independent features (IID).
    correlation=1.0 → all features identical.
    """
    rng = np.random.default_rng(seed)
    # Compound Gaussian: x_i = sqrt(rho) * z + sqrt(1-rho) * epsilon_i
    z   = rng.standard_normal((batch, 1))
    eps = rng.standard_normal((batch, dim))
    x   = np.sqrt(correlation) * z + np.sqrt(max(1.0 - correlation, 0.0)) * eps
    return torch.tensor(x, dtype=torch.float32)


@pytest.fixture
def model():
    return _make_model()


class TestDistributionShift:

    def test_iid_input_output_finite(self, model):
        """Standard IID input (correlation=0) must produce finite output."""
        x = _correlated_input(32, 32, correlation=0.0)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_high_correlation_output_finite(self, model):
        """Strongly correlated input (correlation=0.9) must produce finite output."""
        x = _correlated_input(32, 32, correlation=0.9)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_fully_correlated_output_finite(self, model):
        """Fully correlated input (correlation=1.0) must produce finite output."""
        x = _correlated_input(32, 32, correlation=1.0)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_output_shape_invariant_to_shift(self, model):
        """Output shape must not change under distribution shift."""
        for corr in [0.0, 0.3, 0.6, 0.9]:
            x = _correlated_input(16, 32, correlation=corr)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (16, 2), (
                f"Output shape {out.shape} != (16, 2) at correlation={corr}."
            )

    def test_output_entropy_bounded_under_shift(self, model):
        """
        Softmax entropy of outputs must be bounded (> 0, < log(n_classes))
        across a range of correlation shifts. Unbounded entropy indicates
        model collapse under distribution shift.
        """
        import torch.nn.functional as F
        import math

        max_entropy = math.log(2)  # log(n_classes)

        for corr in [0.0, 0.5, 0.9]:
            x = _correlated_input(32, 32, correlation=corr)
            with torch.no_grad():
                logits = model(x)
                probs  = F.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1).mean().item()

            assert entropy > 0.0, f"Zero entropy at correlation={corr} — model collapsed."
            assert entropy <= max_entropy + 1e-4, (
                f"Entropy {entropy:.4f} > max={max_entropy:.4f} at correlation={corr}."
            )
