"""
src/utils/device_manager.py

Hardware-Agnostic Device Manager.

Auto-detects CUDA / MPS / CPU and provides a unified API so the
rest of the codebase never contains hardcoded .cuda() calls.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class DeviceManager:
    """
    Singleton managing hardware device selection and tensor placement.

    Priority order:
        1. CUDA  - NVIDIA GPUs (RTX, A100, H100)
        2. MPS   - Apple Silicon (M1/M2/M3)
        3. CPU   - Universal fallback

    Usage
    -----
        dm = DeviceManager.get_instance()
        device = dm.get_device()
        model = dm.model_to_device(model)
        tensor = dm.to_device(tensor)
    """

    _instance: Optional["DeviceManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._device: Optional[torch.device] = None
        self._device_info: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls) -> "DeviceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def get_device(self) -> torch.device:
        """Return torch.device for the best available backend."""
        if self._device is not None:
            return self._device

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        return self._device

    def get_device_info(self) -> Dict[str, Any]:
        """Return dictionary with device name, memory, and compute capability."""
        if self._device_info is not None:
            return self._device_info

        device = self.get_device()
        info: Dict[str, Any] = {"device_type": device.type}

        if device.type == "cuda":
            idx = device.index or 0
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "name":                props.name,
                "total_memory_gb":     props.total_memory / 1e9,
                "major":               props.major,
                "minor":               props.minor,
                "multi_processor_count": props.multi_processor_count,
                "cuda_version":        torch.version.cuda,
            })
        elif device.type == "mps":
            info["name"] = "Apple Silicon MPS"
        else:
            import platform
            info["name"] = platform.processor() or "CPU"

        self._device_info = info
        return info

    # ------------------------------------------------------------------
    # Tensor / model placement
    # ------------------------------------------------------------------

    def to_device(
        self,
        obj: Union[torch.Tensor, nn.Module],
        non_blocking: bool = True,
    ) -> Union[torch.Tensor, nn.Module]:
        """Move tensor or module to the detected device."""
        device = self.get_device()
        if isinstance(obj, nn.Module):
            return obj.to(device)
        return obj.to(device, non_blocking=non_blocking)

    def model_to_device(
        self,
        model: nn.Module,
        use_data_parallel: bool = False,
    ) -> nn.Module:
        """
        Move model to device with optional DataParallel wrapping.

        Parameters
        ----------
        model             : PyTorch module to place
        use_data_parallel : Wrap in DataParallel if multiple GPUs available
        """
        device = self.get_device()
        model = model.to(device)

        if use_data_parallel and device.type == "cuda":
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                model = nn.DataParallel(model)

        return model

    # ------------------------------------------------------------------
    # Optimisation settings
    # ------------------------------------------------------------------

    def optimize_for_device(self) -> None:
        """Configure optimal backend settings for the detected device."""
        device = self.get_device()

        if device.type == "cuda":
            # Deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Use deterministic CUDA algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------

    def available_memory_gb(self) -> float:
        """Return available GPU memory in GB (0.0 for CPU/MPS)."""
        device = self.get_device()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            return free / 1e9
        return 0.0

    def empty_cache(self) -> None:
        """Release unused GPU memory."""
        if self.get_device().type == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device})"
 