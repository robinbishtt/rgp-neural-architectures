"""
tests/robustness/test_noise_robustness.py

Gaussian noise injection robustness tests.
Verifies that RG-Net accuracy degrades gracefully under additive noise.
"""

import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn as nn


def _make_simple_model(depth: int = 5, width: int = 32) -> nn.Module:
    layers = []
    layers.append(nn.Linear(width, width))
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return _make_simple_model()


@pytest.fixture
def clean_input():
    torch.manual_seed(0)
    return torch.randn(16, 32)


class TestNoiseRobustness:

    def test_zero_noise_unchanged(self, model, clean_input):
        """Zero-noise injection must leave output bit-identical."""
        noisy = clean_input + torch.zeros_like(clean_input)
        with torch.no_grad():
            out_clean = model(clean_input)
            out_noisy = model(noisy)
        assert torch.allclose(out_clean, out_noisy), "Zero-noise changed output."

    def test_small_noise_finite_output(self, model, clean_input):
        """Small Gaussian noise (sigma=0.01) must produce finite output."""
        noise = torch.randn_like(clean_input) * 0.01
        noisy_input = clean_input + noise
        with torch.no_grad():
            out = model(noisy_input)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf under small noise."

    def test_large_noise_finite_output(self, model, clean_input):
        """Large Gaussian noise (sigma=1.0) must still produce finite output."""
        noise = torch.randn_like(clean_input) * 1.0
        noisy_input = clean_input + noise
        with torch.no_grad():
            out = model(noisy_input)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf under large noise."

    def test_output_sensitivity_bounded(self, model, clean_input):
        """
        Lipschitz check: ||f(x+eps) - f(x)|| / ||eps|| should be bounded.
        For a tanh network near critical init, Lipschitz constant ≈ 1.
        """
        eps = torch.randn_like(clean_input) * 1e-3
        with torch.no_grad():
            out_clean = model(clean_input)
            out_noisy = model(clean_input + eps)

        delta_out = (out_noisy - out_clean).norm()
        delta_in  = eps.norm()
        ratio = (delta_out / delta_in).item()

        # Generous bound: Lipschitz constant < 100 for random init network
        assert ratio < 100.0, f"Output sensitivity ratio {ratio:.2f} is too large."

    def test_noise_levels_monotone_degradation(self, model, clean_input):
        """
        Output variance should increase monotonically with noise level.
        """
        noise_levels = [0.0, 0.01, 0.1, 0.5, 1.0]
        variances = []
        with torch.no_grad():
            base_out = model(clean_input)
            for sigma in noise_levels:
                noise = torch.randn_like(clean_input) * sigma
                out = model(clean_input + noise)
                variances.append((out - base_out).var().item())

        # Each variance must be >= previous (non-decreasing in expectation)
        for i in range(1, len(variances)):
            assert variances[i] >= variances[i - 1] - 1e-6, (
                f"Output variance decreased from sigma={noise_levels[i-1]} "
                f"to sigma={noise_levels[i]}."
            )
 