"""
src/telemetry/jsonl_storage.py

JSONL append-only event log for checkpoint events, errors, and warnings.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict


class JSONLStorage:
    """Append-only JSONL event log. Human-readable and tool-agnostic."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh  = open(self.path, "a")

    def write(self, record: Dict[str, Any]) -> None:
        record.setdefault("timestamp", time.time())
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
