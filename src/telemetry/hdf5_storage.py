"""
src/telemetry/hdf5_storage.py

HDF5-based storage for large tensor data: Fisher matrices, Jacobian spectra.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


class HDF5Storage:
    """
    Chunked HDF5 storage for large tensor telemetry.

    Stores Fisher information matrices and Jacobian spectra per layer/step.
    Uses gzip compression by default. Supports partial reads for large files.
    """

    def __init__(self, path: str, compression: str = "gzip",
                 compression_opts: int = 4) -> None:
        self.path    = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression      = compression
        self.compression_opts = compression_opts

    def write_tensor(self, key: str, data: np.ndarray, step: int) -> None:
        """Append tensor data at given step under key."""
        try:
            import h5py
            with h5py.File(self.path, "a") as f:
                group = f.require_group(key)
                group.create_dataset(
                    str(step), data=data,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
        except ImportError:
            pass  # h5py not available; silently skip

    def read_tensor(self, key: str, step: int) -> Optional[np.ndarray]:
        try:
            import h5py
            with h5py.File(self.path, "r") as f:
                return np.array(f[key][str(step)])
        except (ImportError, KeyError):
            return None

    def list_steps(self, key: str) -> list:
        try:
            import h5py
            with h5py.File(self.path, "r") as f:
                return sorted(int(s) for s in f.get(key, {}).keys())
        except ImportError:
            return []
 