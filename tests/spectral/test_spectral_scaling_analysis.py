"""tests/spectral/test_spectral_scaling_analysis.py"""
import pytest
import torch
import torch.nn as nn


class TestSpectralScalingAnalysis:
    def test_analyze_model_returns_results(self):
        from src.scaling.spectral_scaling import SpectralScalingAnalyzer
        from src.architectures.rg_net.rg_net_standard import RGNetStandard
        torch.manual_seed(0)
        model = RGNetStandard(input_dim=32, n_classes=4)
        analyzer = SpectralScalingAnalyzer()
        results = analyzer.analyze_model(model)
        assert len(results) > 0

    def test_effective_rank_positive(self):
        from src.scaling.spectral_scaling import SpectralScalingAnalyzer
        W = torch.randn(16, 16)
        result = SpectralScalingAnalyzer().analyze_layer(W, layer_index=0)
        assert result.effective_rank > 0
        assert result.max_sv > 0.0

    def test_mp_ks_pval_computed(self):
        from src.scaling.spectral_scaling import SpectralScalingAnalyzer
        W = torch.randn(64, 128) * (1.0 / 128.0) ** 0.5
        result = SpectralScalingAnalyzer().analyze_layer(W, sigma2=1.0 / 128.0)
        assert result.mp_ks_stat is not None
        assert 0.0 <= result.mp_ks_stat <= 1.0
 