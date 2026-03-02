"""
src/utils/telemetry_logger.py

Structured JSONL telemetry logger with multi-backend support.
Backends: TensorBoard, Weights & Biases, MLflow, JSONL file fallback.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """
    Unified telemetry interface logging to one or more backends.

    Usage
    -----
        tl = TelemetryLogger(log_dir="logs/h1_run01", backends=["tensorboard","jsonl"])
        tl.log_scalar("train/loss", 0.45, step=100)
        tl.log_histogram("fisher/eigenvalues", eigenvalues, step=100)
        tl.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        backends: Optional[list] = None,
        experiment_name: str = "rgp_experiment",
    ) -> None:
        self._log_dir        = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._backends       = backends or ["jsonl"]
        self._experiment     = experiment_name
        self._tb_writer      = None
        self._wandb_run      = None
        self._jsonl_path     = self._log_dir / "telemetry.jsonl"
        self._jsonl_file     = open(self._jsonl_path, "a")

        self._init_backends()

    def _init_backends(self) -> None:
        if "tensorboard" in self._backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(str(self._log_dir / "tensorboard"))
            except ImportError:
                logger.warning("TensorBoard not available. Falling back to JSONL.")

        if "wandb" in self._backends:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self._experiment,
                    dir=str(self._log_dir),
                    mode="offline",
                )
            except Exception as exc:
                logger.warning("WandB init failed: %s", exc)

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._write_jsonl({"type": "scalar", "name": name, "value": value, "step": step})
        if self._tb_writer:
            self._tb_writer.add_scalar(name, value, step)
        if self._wandb_run:
            self._wandb_run.log({name: value}, step=step)

    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, torch.Tensor],
        step: int,
    ) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        summary = {
            "mean": float(values.mean()),
            "std":  float(values.std()),
            "min":  float(values.min()),
            "max":  float(values.max()),
        }
        self._write_jsonl({"type": "histogram", "name": name, "step": step, **summary})
        if self._tb_writer:
            self._tb_writer.add_histogram(name, values, step)

    def log_fisher_metric(
        self,
        layer_id: int,
        metric_tensor: torch.Tensor,
        step: int,
    ) -> None:
        ev = torch.linalg.eigvalsh(metric_tensor.detach().cpu()).numpy()
        self.log_histogram(f"fisher/layer_{layer_id}/eigenvalues", ev, step)

    def log_jacobian_spectrum(
        self,
        layer_id: int,
        singular_values: Union[np.ndarray, torch.Tensor],
        step: int,
    ) -> None:
        self.log_histogram(f"jacobian/layer_{layer_id}/singular_values",
                           singular_values, step)

    def log_checkpoint(self, path: str, metadata: Dict[str, Any], step: int) -> None:
        self._write_jsonl({
            "type":     "checkpoint",
            "path":     path,
            "step":     step,
            "metadata": metadata,
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        record["timestamp"] = time.time()
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        self._jsonl_file.close()
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            self._wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
 