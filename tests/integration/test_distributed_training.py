"""tests/integration/test_distributed_training.py"""
import pytest
import torch
import torch.nn as nn
from src.utils.seed_registry import SeedRegistry


def test_distributed_trainer_instantiation():
    from src.training.distributed_trainer import DistributedTrainer
    # Just verify instantiation without actual DDP (no NCCL in CI)
    trainer = DistributedTrainer(rank=0, world_size=1)
    assert trainer.rank == 0
    assert trainer.is_main_rank()
 