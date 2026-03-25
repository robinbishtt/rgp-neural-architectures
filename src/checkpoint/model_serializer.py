from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
class ModelStateSerializer:
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: Path,
    ) -> None:
        path = Path(path)
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(model_state, path / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), path / "optimizer.pt")
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: Path,
        map_location: str = "cpu",
    ) -> None:
        path = Path(path)
        state_dict = torch.load(path / "model.pt", map_location=map_location)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state_dict)
        if optimizer is not None and (path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=map_location))