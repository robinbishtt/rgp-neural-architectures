import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
import torch
def test_mixed_precision_trainer_instantiation():
    from src.training.mixed_precision_trainer import MixedPrecisionTrainer
    trainer = MixedPrecisionTrainer(dtype=torch.float32)
    assert trainer is not None