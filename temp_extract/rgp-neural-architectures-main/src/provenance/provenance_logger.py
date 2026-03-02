"""
src/provenance/provenance_logger.py

ProvenanceLogger: immutable, append-only provenance trail for all
data generation and training events.

Motivation
----------
The reproducibility guarantee for this research codebase requires
that any reviewer can reconstruct the exact sequence of operations that
produced any figure or table entry.  The standard approach of logging
to console is insufficient: console logs are ephemeral, non-structured,
and cannot be queried programmatically.

ProvenanceLogger writes a structured, append-only JSONL provenance trail
that captures every significant system event:
    * Dataset generation: seed, parameters, SHA-256 checksum
    * Model initialisation: architecture, sigma_w, sigma_b, seed
    * Training runs: hyperparameters, epoch, timestamp
    * Evaluation: dataset, accuracy, correlation lengths
    * Figure generation: input data files, output file, git commit

Each record includes a microsecond-precision ISO 8601 timestamp, the
Python process ID, the hostname, and the SHA-256 of any relevant files.
This allows post-hoc reconstruction of the exact computational provenance
for any published result.

The provenance trail is stored at: results/provenance/provenance.jsonl
A separate index file (provenance_index.json) enables efficient lookup
by event type, experiment ID, or timestamp range.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROVENANCE_DIR = Path("results") / "provenance"


class ProvenanceRecord:
    """A single immutable provenance event."""

    def __init__(
        self,
        event_type:   str,
        experiment_id: str,
        details:       Dict[str, Any],
        file_hashes:   Optional[Dict[str, str]] = None,
    ) -> None:
        self.event_type    = event_type
        self.experiment_id = experiment_id
        self.details       = details
        self.file_hashes   = file_hashes or {}
        self.timestamp     = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        self.pid           = os.getpid()
        self.hostname      = socket.gethostname()

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            event_type    = self.event_type,
            experiment_id = self.experiment_id,
            timestamp     = self.timestamp,
            pid           = self.pid,
            hostname      = self.hostname,
            file_hashes   = self.file_hashes,
            details       = self.details,
        )


class ProvenanceLogger:
    """
    Append-only provenance logger for RGP experiments.

    Parameters
    ----------
    log_dir : Path or str
        Directory where provenance.jsonl and provenance_index.json are written.
    experiment_id : str
        Unique experiment identifier.  Included in every record for filtering.

    Example
    -------
    ::
        plog = ProvenanceLogger(log_dir="results/provenance", experiment_id="h1_run01")
        plog.log_dataset_generated(
            dataset_path=Path("data/hierarchical_mnist"),
            seed=42, n_samples=50000,
        )
        plog.log_training_start(
            model_config={"depth": 100, "width": 256},
            training_config={"lr": 1e-3, "epochs": 100},
        )
    """

    def __init__(
        self,
        log_dir:       Any = None,
        experiment_id: str = "default",
    ) -> None:
        if log_dir is None:
            log_dir = _DEFAULT_PROVENANCE_DIR
        self.log_dir       = Path(log_dir)
        self.experiment_id = experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path  = self.log_dir / "provenance.jsonl"
        self._index_path  = self.log_dir / "provenance_index.json"
        self._jsonl_file  = open(self._jsonl_path, "a", encoding="utf-8")
        self._index: List[Dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # High-level event loggers
    # ------------------------------------------------------------------

    def log_dataset_generated(
        self,
        dataset_path: Path,
        seed:         int,
        n_samples:    int,
        parameters:   Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a dataset generation event with SHA-256 checksum."""
        checksum = self._hash_path(dataset_path)
        self._write(ProvenanceRecord(
            event_type    = "dataset_generated",
            experiment_id = self.experiment_id,
            details       = dict(dataset_path=str(dataset_path), seed=seed,
                                  n_samples=n_samples, parameters=parameters or {}),
            file_hashes   = {str(dataset_path): checksum},
        ))

    def log_model_init(
        self,
        model_config: Dict[str, Any],
        seed:         int,
    ) -> None:
        """Record model initialisation with architecture parameters."""
        self._write(ProvenanceRecord(
            event_type    = "model_initialised",
            experiment_id = self.experiment_id,
            details       = dict(model_config=model_config, seed=seed),
        ))

    def log_training_start(
        self,
        model_config:    Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> None:
        """Record training run start."""
        self._write(ProvenanceRecord(
            event_type    = "training_start",
            experiment_id = self.experiment_id,
            details       = dict(model=model_config, training=training_config),
        ))

    def log_training_complete(
        self,
        final_metrics:   Dict[str, float],
        elapsed_seconds: float,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """Record training completion with final metrics."""
        fhashes = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            fhashes[str(checkpoint_path)] = self._hash_path(Path(checkpoint_path))

        self._write(ProvenanceRecord(
            event_type    = "training_complete",
            experiment_id = self.experiment_id,
            details       = dict(metrics=final_metrics, elapsed_s=elapsed_seconds,
                                  checkpoint=str(checkpoint_path)),
            file_hashes   = fhashes,
        ))

    def log_figure_generated(
        self,
        figure_path: Path,
        input_files: Optional[List[Path]] = None,
        git_commit:  Optional[str] = None,
    ) -> None:
        """Record figure generation with input data checksums."""
        fhashes = {}
        for f in (input_files or []):
            if Path(f).exists():
                fhashes[str(f)] = self._hash_path(Path(f))
        if Path(figure_path).exists():
            fhashes[str(figure_path)] = self._hash_path(Path(figure_path))

        self._write(ProvenanceRecord(
            event_type    = "figure_generated",
            experiment_id = self.experiment_id,
            details       = dict(figure=str(figure_path), git_commit=git_commit),
            file_hashes   = fhashes,
        ))

    def log_custom(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a custom provenance event."""
        self._write(ProvenanceRecord(
            event_type    = event_type,
            experiment_id = self.experiment_id,
            details       = details,
        ))

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _write(self, record: ProvenanceRecord) -> None:
        d = record.to_dict()
        self._jsonl_file.write(json.dumps(d) + "\n")
        self._jsonl_file.flush()
        self._update_index(d)

    def _update_index(self, record: Dict[str, Any]) -> None:
        entry = {k: record[k] for k in ("event_type", "experiment_id", "timestamp")}
        self._index.append(entry)
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=1)

    def _load_index(self) -> List[Dict[str, Any]]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return []

    @staticmethod
    def _hash_path(path: Path, chunk_size: int = 1 << 20) -> str:
        if path.is_file():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
            return h.hexdigest()
        if path.is_dir():
            h = hashlib.sha256()
            for fp in sorted(path.rglob("*")):
                if fp.is_file():
                    h.update(str(fp.relative_to(path)).encode())
                    # hash the file content too
                    with open(fp, "rb") as f:
                        while chunk := f.read(chunk_size):
                            h.update(chunk)
            return h.hexdigest()
        return "missing"

    def close(self) -> None:
        """Close the underlying JSONL file handle."""
        self._jsonl_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
