"""
tests/conftest.py

Shared pytest fixtures for all test suites.
Torch fixtures are skipped automatically when torch is not available.
"""
import pytest
import numpy as np

# Torch is optional - tests requiring it are skipped when unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset global seed to 42 before every test for reproducibility."""
    import random
    random.seed(42)
    np.random.seed(42)
    try:
        from src.utils.seed_registry import SeedRegistry
        SeedRegistry.get_instance().set_master_seed(42)
    except Exception:
        pass
    yield


@pytest.fixture
def device():
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    try:
        from src.utils.device_manager import DeviceManager
        return DeviceManager.get_instance().get_device()
    except Exception:
        return torch.device("cpu")


@pytest.fixture
def small_linear_model():
    """Tiny linear model for fast unit testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    return model


@pytest.fixture
def random_psd_matrix():
    """Return small random PSD matrix for Fisher tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    n = 8
    A = torch.randn(n, n)
    return A @ A.T + torch.eye(n) * 0.1


@pytest.fixture
def synthetic_dataset():
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    from src.datasets.synthetic_hierarchy import SyntheticHierarchy
    return SyntheticHierarchy(n_samples=100, n_features=64, seed=42)
