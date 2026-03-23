"""
src/telemetry/parquet_storage.py

Parquet storage for tabular metrics: loss curves, accuracy, hyperparameters.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List


class ParquetStorage:
    """
    Columnar Parquet storage for training metrics.

    Buffers records in memory and flushes to Parquet periodically.
    Files are queryable with pandas for offline analysis.
    """

    def __init__(self, path: str, flush_every: int = 100) -> None:
        self.path        = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        try:
            import pandas as pd
            df_new = pd.DataFrame(self._buffer)
            if self.path.exists():
                df_old = pd.read_parquet(self.path)
                df     = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_parquet(self.path, index=False)
            self._buffer.clear()
        except ImportError:
            pass

    def close(self) -> None:
        self.flush()
 