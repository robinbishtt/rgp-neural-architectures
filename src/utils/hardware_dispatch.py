from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
@dataclass
class HardwareCapabilities:
    device_type: str           
    device_name: str
    supports_fp16: bool
    supports_bf16: bool
    supports_tf32: bool
    n_gpus: int
    vram_gb: float
    compute_capability: Optional[tuple]   
def detect_hardware() -> HardwareCapabilities:
    if torch.cuda.is_available():
        idx   = 0
        props = torch.cuda.get_device_properties(idx)
        cc    = (props.major, props.minor)
        return HardwareCapabilities(
            device_type          = "cuda",
            device_name          = props.name,
            supports_fp16        = True,
            supports_bf16        = cc >= (8, 0),   
            supports_tf32        = cc >= (8, 0),
            n_gpus               = torch.cuda.device_count(),
            vram_gb              = props.total_memory / 1e9,
            compute_capability   = cc,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareCapabilities(
            device_type         = "mps",
            device_name         = "Apple Silicon",
            supports_fp16       = True,
            supports_bf16       = False,
            supports_tf32       = False,
            n_gpus              = 0,
            vram_gb             = 0.0,
            compute_capability  = None,
        )
    else:
        return HardwareCapabilities(
            device_type         = "cpu",
            device_name         = "CPU",
            supports_fp16       = False,
            supports_bf16       = False,
            supports_tf32       = False,
            n_gpus              = 0,
            vram_gb             = 0.0,
            compute_capability  = None,
        )
def select_dtype(
    hw: HardwareCapabilities,
    prefer_bf16: bool = True,
) -> torch.dtype:
    if prefer_bf16 and hw.supports_bf16:
        return torch.bfloat16
    if hw.supports_fp16:
        return torch.float16
    return torch.float32