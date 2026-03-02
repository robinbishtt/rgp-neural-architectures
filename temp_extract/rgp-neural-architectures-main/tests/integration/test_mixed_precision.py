"""tests/integration/test_mixed_precision.py"""
import pytest
import torch
import torch.nn as nn
from src.utils.seed_registry import SeedRegistry


def test_mixed_precision_trainer_instantiation():
    from src.training.mixed_precision_trainer import MixedPrecisionTrainer
    trainer = MixedPrecisionTrainer(dtype=torch.float32)
    assert trainer is not None
