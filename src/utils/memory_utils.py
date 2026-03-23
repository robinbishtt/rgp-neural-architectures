"""
src/utils/memory_utils.py

Memory safety utilities: estimation, tracking, and gradient checkpointing helpers.
"""

from __future__ import annotations

import contextlib
from typing import Generator, Optional

import torch
import torch.nn as nn


def estimate_model_memory_gb(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    include_gradients: bool = True,
) -> float:
    """
    Estimate GPU memory required for model forward+backward pass.

    Returns estimated memory in GB.
    """
    bytes_per_element = torch.finfo(dtype).bits // 8

    # Parameter memory
    n_params = sum(p.numel() for p in model.parameters())
    param_mem = n_params * bytes_per_element
    if include_gradients:
        param_mem *= 2  # grad buffers

    # Activation memory (rough estimate: 2x parameter memory for deep nets)
    activation_mem = n_params * bytes_per_element * 2

    # Input tensor
    input_numel = batch_size * int(torch.prod(torch.tensor(input_shape)).item())
    input_mem = input_numel * bytes_per_element

    total_bytes = param_mem + activation_mem + input_mem
    return total_bytes / 1e9


@contextlib.contextmanager
def memory_guard(
    threshold_gb: float = 1.0,
    device: Optional[torch.device] = None,
) -> Generator[None, None, None]:
    """
    Context manager that checks available GPU memory before and after a block.
    Logs a warning if memory usage exceeds threshold_gb.
    """
    import logging
    logger = logging.getLogger(__name__)

    if device is None or device.type != "cuda":
        yield
        return

    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()

    yield

    torch.cuda.synchronize()
    free_after, total = torch.cuda.mem_get_info()
    used_gb = (free_before - free_after) / 1e9

    if used_gb > threshold_gb:
        logger.warning("Block consumed %.2f GB GPU memory.", used_gb)


class GradientCheckpointHelper:
    """
    Wraps nn.Module layers with gradient checkpointing for memory reduction.
    Reduces activation memory by ~60% at cost of ~25% extra compute.
    """

    @staticmethod
    def wrap(model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing on all Sequential / ModuleList children."""
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
        return model

    @staticmethod
    def checkpoint_sequential(
        functions: list,
        segments: int,
        input_tensor: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply torch.utils.checkpoint.checkpoint_sequential."""
        import torch.utils.checkpoint as cp
        return cp.checkpoint_sequential(functions, segments, input_tensor, **kwargs)
 