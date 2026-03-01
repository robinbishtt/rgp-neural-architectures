"""
src/training/distributed_trainer.py

Multi-GPU distributed training with synchronous SGD via torch.distributed.
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training.trainer import Trainer


class DistributedTrainer(Trainer):
    """
    Extends Trainer for multi-GPU distributed training (DDP).

    Handles: process group initialization, DDP wrapping, gradient
    synchronization, and checkpoint coordination across ranks.
    """

    def __init__(self, rank: int, world_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rank       = rank
        self.world_size = world_size

    def setup_process_group(self, backend: str = "nccl") -> None:
        dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def wrap_model_ddp(self, model: nn.Module) -> DDP:
        return DDP(model.to(self.rank), device_ids=[self.rank])

    def cleanup(self) -> None:
        dist.destroy_process_group()

    def is_main_rank(self) -> bool:
        return self.rank == 0
