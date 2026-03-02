"""
src/utils/seed_registry.py

Global Determinism Controller — Singleton managing all random seeds.

In a 340+ file system, any single call to random.seed() at the wrong time
corrupts the L_min ~ log(xi_data) data collapse. This module ensures that
NO module generates its own seed; all RNG access routes through here.
"""

from __future__ import annotations

import random
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch


class SeedRegistry:
    """
    Thread-safe Singleton controlling all random number generators.

    Usage
    -----
        reg = SeedRegistry.get_instance()
        reg.set_master_seed(42)
        seed = reg.get_worker_seed(worker_id=3)
        state = reg.snapshot_state()
        # ... later ...
        reg.restore_state(state)
    """

    _instance: Optional["SeedRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._master_seed: Optional[int] = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SeedRegistry":
        """Return the Singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Seed management
    # ------------------------------------------------------------------

    def set_master_seed(self, seed: int) -> None:
        """
        Set the master seed and propagate to all RNGs.

        Affects: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
        """
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")

        self._master_seed = seed
        self._step = 0

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic CUDA ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_worker_seed(self, worker_id: int) -> int:
        """
        Generate a deterministic worker-specific seed.

        Uses: hash(master_seed || worker_id) to guarantee uniqueness.
        """
        if self._master_seed is None:
            raise RuntimeError(
                "SeedRegistry.set_master_seed() must be called before get_worker_seed()."
            )
        combined = (self._master_seed * 2654435761 + worker_id * 1013904223) & 0xFFFFFFFF
        return int(combined)

    def advance(self, n: int = 1) -> None:
        """Advance the step counter (for step-aware reproducibility)."""
        self._step += n

    @property
    def master_seed(self) -> Optional[int]:
        return self._master_seed

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # State snapshot / restore
    # ------------------------------------------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        """
        Capture complete RNG state for checkpoint resume.

        Returns a dictionary containing all RNG states. Pass this dict
        to CheckpointManager to embed it in checkpoint files.
        """
        state: Dict[str, Any] = {
            "master_seed":  self._master_seed,
            "step":         self._step,
            "python_state": random.getstate(),
            "numpy_state":  np.random.get_state(),
            "torch_state":  torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_states"] = [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ]
        return state

    def restore_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore all RNG states from a checkpoint dictionary.

        This guarantees bit-exact continuation from any saved state.
        """
        self._master_seed = state_dict["master_seed"]
        self._step        = state_dict["step"]

        random.setstate(state_dict["python_state"])
        np.random.set_state(state_dict["numpy_state"])
        torch.set_rng_state(state_dict["torch_state"])

        if torch.cuda.is_available() and "cuda_states" in state_dict:
            for i, s in enumerate(state_dict["cuda_states"]):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(s, device=i)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def worker_init_fn(self, worker_id: int) -> None:
        """
        DataLoader worker_init_fn that seeds each worker deterministically.

        Usage:
            DataLoader(..., worker_init_fn=SeedRegistry.get_instance().worker_init_fn)
        """
        seed = self.get_worker_seed(worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __repr__(self) -> str:
        return (
            f"SeedRegistry(master_seed={self._master_seed}, step={self._step})"
        )
 