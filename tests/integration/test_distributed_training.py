import pytest
torch = pytest.importorskip("torch", reason="torch not installed")


def test_distributed_trainer_instantiation():
    from src.training.distributed_trainer import DistributedTrainer
    trainer = DistributedTrainer(rank=0, world_size=1)
    assert trainer.rank == 0
    assert trainer.is_main_rank()