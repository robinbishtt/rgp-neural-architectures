"""
src/utils/provenance.py

SHA-256 data provenance — verifies dataset integrity before training begins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from src.utils.error_handler import DataIntegrityError


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class DataAuditor:
    """
    Verifies SHA-256 checksums of datasets before training.

    Steps executed automatically before any training run:
        1. Dataset generated with deterministic seed from SeedRegistry
        2. compute_checksum() calculates SHA-256
        3. verify_checksum() compares against master hash
        4. Training gates on checksum match
    """

    def compute_checksum(self, dataset_path: Path) -> str:
        """Compute SHA-256 of the entire dataset directory."""
        dataset_path = Path(dataset_path)
        if dataset_path.is_file():
            return compute_file_sha256(dataset_path)

        h = hashlib.sha256()
        for fpath in sorted(dataset_path.rglob("*")):
            if fpath.is_file():
                h.update(str(fpath.relative_to(dataset_path)).encode())
                h.update(compute_file_sha256(fpath).encode())
        return h.hexdigest()

    def verify_checksum(self, dataset_path: Path, expected_hash: str) -> bool:
        """
        Compare computed checksum against expected. Raises DataIntegrityError on mismatch.
        """
        actual = self.compute_checksum(Path(dataset_path))
        if actual != expected_hash:
            raise DataIntegrityError(
                f"Checksum mismatch for {dataset_path}.\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual}\n"
                "Dataset may be corrupted or generated with different parameters."
            )
        return True

    def generate_manifest(self, dataset_dir: Path) -> Dict[str, str]:
        """Create manifest dict: relative_path -> sha256."""
        dataset_dir = Path(dataset_dir)
        manifest: Dict[str, str] = {}
        for fpath in sorted(dataset_dir.rglob("*")):
            if fpath.is_file():
                rel = str(fpath.relative_to(dataset_dir))
                manifest[rel] = compute_file_sha256(fpath)
        return manifest

    def save_manifest(self, dataset_dir: Path, manifest_path: Path) -> None:
        manifest = self.generate_manifest(dataset_dir)
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    def verify_manifest(self, manifest_path: Path) -> bool:
        """Verify all files in a manifest match their recorded checksums."""
        manifest_path = Path(manifest_path)
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        base_dir = manifest_path.parent
        for rel_path, expected_hash in manifest.items():
            fpath = base_dir / rel_path
            if not fpath.exists():
                raise DataIntegrityError(f"Missing file: {fpath}")
            actual = compute_file_sha256(fpath)
            if actual != expected_hash:
                raise DataIntegrityError(
                    f"Checksum mismatch: {rel_path}\n"
                    f"  Expected: {expected_hash}\n  Actual: {actual}"
                )
        return True
