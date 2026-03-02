"""
tests/unit/test_fisher_monte_carlo.py

Unit tests for FisherMonteCarloEstimator.

Tests verify:
  1. estimate_diagonal returns non-negative values (Fisher is PSD)
  2. estimate_trace returns a positive float (Tr(F) > 0 for any trainable network)
  3. estimate_layer_metric returns a square, symmetric matrix
  4. Inherited FisherMetricBase methods work: condition_number, effective_rank,
     is_positive_semidefinite
  5. compute_all_layers returns one metric per linear layer
"""
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def small_model():
    """A minimal two-layer MLP for deterministic testing."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.Tanh(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return torch.randn(16, 8)


@pytest.fixture
def targets():
    return torch.randint(0, 4, (16,))


@pytest.fixture
def loss_fn():
    return nn.CrossEntropyLoss()


class TestFisherMonteCarloEstimatorBasic:
    def test_import(self):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        assert FisherMonteCarloEstimator is not None

    def test_instantiation_default(self):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator()
        assert est.n_samples == 100
        assert est.estimator == "hutchinson"

    def test_instantiation_custom(self):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=50, estimator="rademacher")
        assert est.n_samples == 50

    def test_bad_estimator_raises(self):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        with pytest.raises(ValueError):
            FisherMonteCarloEstimator(estimator="unknown_method")


class TestFisherDiagonalEstimate:
    def test_diagonal_nonnegative(self, small_model, inputs, targets, loss_fn):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=10)
        diag = est.estimate_diagonal(small_model, loss_fn, inputs, targets)
        assert (diag >= 0).all(), "Fisher diagonal must be non-negative everywhere"

    def test_diagonal_length(self, small_model, inputs, targets, loss_fn):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=5)
        diag = est.estimate_diagonal(small_model, loss_fn, inputs, targets)
        n_params = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
        assert diag.shape == (n_params,), (
            f"Expected diagonal length {n_params}, got {diag.shape[0]}"
        )

    def test_diagonal_is_tensor(self, small_model, inputs, targets, loss_fn):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=5)
        diag = est.estimate_diagonal(small_model, loss_fn, inputs, targets)
        assert isinstance(diag, torch.Tensor)


class TestFisherTraceEstimate:
    def test_trace_is_float(self, small_model, inputs, targets, loss_fn):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=10)
        tr = est.estimate_trace(small_model, loss_fn, inputs, targets)
        assert isinstance(tr, float)

    def test_trace_finite(self, small_model, inputs, targets, loss_fn):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=10)
        tr = est.estimate_trace(small_model, loss_fn, inputs, targets)
        assert torch.isfinite(torch.tensor(tr)), "Trace estimate must be finite"


class TestLayerMetricEstimate:
    def test_layer_metric_shape_square(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        assert g.shape[0] == g.shape[1], "Layer metric must be a square matrix"

    def test_layer_metric_symmetric(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        assert torch.allclose(g, g.t(), atol=1e-6), "Layer metric must be symmetric"

    def test_layer_metric_layer0_dim(self, small_model, inputs):
        """Layer 0 (Linear(8,16)): metric should be (8,8) in input space."""
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        assert g.shape == (8, 8)

    def test_layer_metric_layer1_dim(self, small_model, inputs):
        """Layer 1 (Linear(16,4)): metric should be (16,16) in input space."""
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=1)
        assert g.shape == (16, 16)

    def test_invalid_layer_idx_raises(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=5)
        with pytest.raises(IndexError):
            est.estimate_layer_metric(small_model, inputs, layer_idx=99)


class TestFisherMetricBaseInheritance:
    """Verify that FisherMonteCarloEstimator correctly inherits FisherMetricBase."""

    def test_inherits_fisher_metric_base(self):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        from src.core.fisher.fisher_base import FisherMetricBase
        assert issubclass(FisherMonteCarloEstimator, FisherMetricBase)

    def test_condition_number_ge_one(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        kappa = est.condition_number(g)
        assert kappa >= 1.0 - 1e-6, f"Condition number {kappa} < 1"

    def test_effective_rank_positive(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        rank = est.effective_rank(g)
        assert rank >= 1, f"Effective rank {rank} must be >= 1"

    def test_effective_rank_le_matrix_size(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        rank = est.effective_rank(g)
        assert rank <= g.shape[0], "Effective rank cannot exceed matrix dimension"

    def test_is_psd_returns_bool(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        g = est.estimate_layer_metric(small_model, inputs, layer_idx=0)
        result = est.is_positive_semidefinite(g)
        assert isinstance(result, bool)

    def test_compute_all_layers_count(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        metrics = est.compute_all_layers(small_model, inputs)
        n_linear = sum(1 for m in small_model.modules() if isinstance(m, nn.Linear))
        assert len(metrics) == n_linear, (
            f"Expected {n_linear} layer metrics, got {len(metrics)}"
        )

    def test_compute_all_layers_each_square(self, small_model, inputs):
        from src.core.fisher.monte_carlo import FisherMonteCarloEstimator
        est = FisherMonteCarloEstimator(n_samples=8)
        metrics = est.compute_all_layers(small_model, inputs)
        for i, g in enumerate(metrics):
            assert g.shape[0] == g.shape[1], f"Layer {i} metric is not square"
