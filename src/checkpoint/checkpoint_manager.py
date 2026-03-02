"""
src/checkpoint/checkpoint_manager.py

CheckpointManager: orchestrates all checkpoint save/load operations.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Orchestrates checkpoint save/load with configurable frequency.

    Checkpoint directory structure:
        checkpoints/
          checkpoint_0100/
            model.pt
            optimizer.pt
            rng_state.pkl
            metrics.json
            config.yaml
          checkpoint_best/   <- symlink or copy to best checkpoint
          latest             <- text file with latest checkpoint path
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every_n_steps: int = 1000,
        keep_last_n: int = 3,
    ) -> None:
        self.checkpoint_dir     = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n        = keep_last_n
        self._checkpoints: list = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save complete checkpoint at given step."""
        from src.checkpoint.model_serializer import ModelStateSerializer
        from src.checkpoint.rng_serializer import RNGStateSerializer
        from src.checkpoint.metric_serializer import MetricStateSerializer

        ckpt_path = self.checkpoint_dir / f"checkpoint_{step:08d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        ModelStateSerializer().save(model, optimizer, ckpt_path)
        RNGStateSerializer().save(ckpt_path)
        MetricStateSerializer().save(metrics or {}, ckpt_path)

        if config:
            import yaml
            with open(ckpt_path / "config.yaml", "w") as f:
                yaml.dump(config, f)

        # Update latest pointer
        with open(self.checkpoint_dir / "latest", "w") as f:
            f.write(str(ckpt_path))

        if is_best:
            best = self.checkpoint_dir / "checkpoint_best"
            if best.exists():
                shutil.rmtree(best)
            shutil.copytree(ckpt_path, best)

        self._checkpoints.append(ckpt_path)
        self._prune_old_checkpoints()
        return ckpt_path

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint. Defaults to latest if no path given."""
        from src.checkpoint.model_serializer import ModelStateSerializer
        from src.checkpoint.rng_serializer import RNGStateSerializer
        from src.checkpoint.metric_serializer import MetricStateSerializer

        if checkpoint_path is None:
            latest_file = self.checkpoint_dir / "latest"
            if not latest_file.exists():
                raise FileNotFoundError("No checkpoints found.")
            checkpoint_path = latest_file.read_text().strip()

        ckpt_path = Path(checkpoint_path)
        state = {}
        if model is not None:
            ModelStateSerializer().load(model, optimizer, ckpt_path)
        RNGStateSerializer().load(ckpt_path)
        state["metrics"] = MetricStateSerializer().load(ckpt_path)
        return state

    def _prune_old_checkpoints(self) -> None:
        if len(self._checkpoints) > self.keep_last_n:
            to_delete = self._checkpoints[:-self.keep_last_n]
            for path in to_delete:
                if path.exists() and path.name != "checkpoint_best":
                    shutil.rmtree(path)
            self._checkpoints = self._checkpoints[-self.keep_last_n:]

    def should_save(self, step: int) -> bool:
        return step % self.save_every_n_steps == 0
 