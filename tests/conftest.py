"""
tests/conftest.py

Shared pytest fixtures for all test suites.
"""
import pytest
import numpy as np
import torch
from src.utils.seed_registry import SeedRegistry
from src.utils.device_manager import DeviceManager


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset global seed to 42 before every test for reproducibility."""
    SeedRegistry.get_instance().set_master_seed(42)
    yield


@pytest.fixture
def device():
    return DeviceManager.get_instance().get_device()


@pytest.fixture
def small_linear_model():
    """Tiny linear model for fast unit testing."""
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    return model


@pytest.fixture
def random_psd_matrix():
    """Return small random PSD matrix for Fisher tests."""
    n = 8
    A = torch.randn(n, n)
    return A @ A.T + torch.eye(n) * 0.1


@pytest.fixture
def synthetic_dataset():
    from src.datasets.synthetic_hierarchy import SyntheticHierarchy
    return SyntheticHierarchy(n_samples=100, n_features=64, seed=42)
 