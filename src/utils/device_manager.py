from __future__ import annotations
import threading
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
class DeviceManager:
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
    def get_device(self) -> torch.device:
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
        if self._device_info is not None:
            return self._device_info
        device = self.get_device()
        info: Dict[str, Any] = {"device_type": device.type}
        if device.type == "cuda":
            idx = device.index or 0
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "name":            props.name,
                "memory_gb":       props.total_memory / 1e9,
                "compute_major":   props.major,
                "compute_minor":   props.minor,
                "n_multiprocessors": props.multi_processor_count,
                "cuda_version":    torch.version.cuda,
            })
        elif device.type == "mps":
            info["name"] = "Apple Silicon MPS"
        else:
            import platform
            info["name"] = platform.processor() or "CPU"
        self._device_info = info
        return info
    def to_device(
        self,
        obj: Union[torch.Tensor, nn.Module],
        non_blocking: bool = True,
    ) -> Union[torch.Tensor, nn.Module]:
        device = self.get_device()
        if isinstance(obj, nn.Module):
            return obj.to(device)
        return obj.to(device, non_blocking=non_blocking)
    def model_to_device(
        self,
        model: nn.Module,
        use_data_parallel: bool = False,
    ) -> nn.Module:
        device = self.get_device()
        model = model.to(device)
        if use_data_parallel and device.type == "cuda":
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                model = nn.DataParallel(model)
        return model
    def optimize_for_device(self) -> None:
        device = self.get_device()
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    def available_memory_gb(self) -> float:
        device = self.get_device()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            return free / 1e9
        return 0.0
    def empty_cache(self) -> None:
        if self.get_device().type == "cuda":
            torch.cuda.empty_cache()
    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device})"