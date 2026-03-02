"""tests/integration/test_mixed_precision.py"""
import torch


def test_mixed_precision_trainer_instantiation():
    from src.training.mixed_precision_trainer import MixedPrecisionTrainer
    trainer = MixedPrecisionTrainer(dtype=torch.float32)
    assert trainer is not None
 