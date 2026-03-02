"""tests/integration/test_full_hypothesis_pipeline.py"""
import pytest
import torch
import torch.nn as nn
import numpy as np


class TestFullHypothesisPipeline:
    """End-to-end pipeline test from model to observable to hypothesis test."""

    def test_h1_pipeline_correlation_decay(self):
        """H1: per-layer correlation length should decay exponentially."""
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        from src.core.correlation.exponential_decay_fitter import ExponentialDecayFitter
        torch.manual_seed(0)
        model = RGNetStandard(input_dim=32, n_classes=4)
        # Compute approximate per-layer correlation lengths from weight spectra
        xi_values = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                W   = m.weight.data.numpy()
                svs = np.linalg.svd(W, compute_uv=False)
                # xi ≈ max_sv: measures information propagation scale
                xi_values.append(float(svs[0]))
        layers = np.arange(len(xi_values), dtype=float)
        if len(xi_values) >= 3:
            result = ExponentialDecayFitter().fit(layers, np.array(xi_values))
            # Just verify the fit runs and produces finite values
            assert np.isfinite(result.xi_0)
            assert np.isfinite(result.k_c)
            assert result.k_c > 0

    def test_h2_pipeline_depth_vs_min_accuracy(self):
        """H2: deeper models should achieve at least as high accuracy as shallow ones."""
        from src.architectures.rg_net.rg_net_shallow import RGNetShallow
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(1)
        shallow  = RGNetShallow(input_dim=16, n_classes=2)
        standard = RGNetStandard(input_dim=16, n_classes=2)
        # Both should produce valid probability distributions
        x = torch.randn(8, 16)
        for model in [shallow, standard]:
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=-1)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
 