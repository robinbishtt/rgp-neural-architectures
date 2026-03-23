"""
src/provenance/checksum_registry.py

ChecksumRegistry: versioned SHA-256 registry for all reproducible assets.

Relationship to master_hashes.py
---------------------------------
master_hashes.py contains the hard-coded, version-controlled SHA-256 hashes
for the *official* datasets described in the paper.  ChecksumRegistry extends
this with a *runtime-updatable* registry that also tracks:
    * Intermediate data files (cached embeddings, pre-processed datasets)
    * Generated figure outputs (PDF, PNG)
    * Model checkpoint files
    * Result JSON files from each experiment

The registry is stored as a JSON file (results/provenance/checksum_registry.json)
and is updated whenever a new asset is registered via register().  On load,
the registry is merged with master_hashes.py so that both official and runtime
assets can be verified through a unified verify() API.

This is used by the CI/CD pipeline (Makefile validate target) to detect any
unintended changes to tracked assets between experiment runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("results") / "provenance" / "checksum_registry.json"


class ChecksumRegistry:
    """
    Runtime-updatable SHA-256 registry for tracked experiment assets.

    Parameters
    ----------
    registry_path : Path or str, optional
        Location of the persistent registry JSON file.
        Defaults to results/provenance/checksum_registry.json.
    merge_master : bool
        If True, merges master_hashes.py entries into the registry on load.

    Example
    -------
    ::
        reg = ChecksumRegistry()
        reg.register("data/hierarchical_mnist", path=Path("data/hierarchical_mnist"))
        is_ok = reg.verify("data/hierarchical_mnist", Path("data/hierarchical_mnist"))
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        merge_master:  bool = True,
    ) -> None:
        self._path   = Path(registry_path or _DEFAULT_REGISTRY_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, str] = self._load()

        if merge_master:
            self._merge_master_hashes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        asset_id:  str,
        path:      Path,
        overwrite: bool = False,
    ) -> str:
        """
        Compute SHA-256 of ``path`` and register it under ``asset_id``.

        Parameters
        ----------
        asset_id : str
            Logical identifier for the asset (e.g., "dataset:hierarchical_mnist_v1").
        path : Path
            File or directory to hash.
        overwrite : bool
            If False and asset_id is already registered, raises ValueError.

        Returns
        -------
        str
            The computed SHA-256 hex digest.
        """
        if asset_id in self._store and not overwrite:
            raise ValueError(
                f"Asset '{asset_id}' already registered.  "
                "Pass overwrite=True to replace the existing hash."
            )
        checksum = self._hash(Path(path))
        self._store[asset_id] = checksum
        self._save()
        logger.info("Registered asset '%s' with SHA-256 %s…", asset_id, checksum[:12])
        return checksum

    def verify(
        self,
        asset_id: str,
        path:     Path,
    ) -> bool:
        """
        Verify that ``path`` matches the registered hash for ``asset_id``.

        Parameters
        ----------
        asset_id : str
            Registered logical identifier.
        path : Path
            Current file or directory to verify.

        Returns
        -------
        bool
            True if hashes match.

        Raises
        ------
        KeyError
            If ``asset_id`` is not registered.
        DataIntegrityError
            If hashes do not match.
        """
        if asset_id not in self._store:
            raise KeyError(
                f"Asset '{asset_id}' not found in registry.  "
                f"Run register() first or check master_hashes.py."
            )
        expected = self._store[asset_id]
        actual   = self._hash(Path(path))

        if actual != expected:
            from src.utils.error_handler import DataIntegrityError
            raise DataIntegrityError(
                f"Checksum mismatch for asset '{asset_id}' at {path}.\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}\n"
                "The file may be corrupted or was generated with different parameters."
            )
        logger.debug("Verified '%s' ✓", asset_id)
        return True

    def get(self, asset_id: str) -> Optional[str]:
        """Return the registered hash for ``asset_id``, or None if not found."""
        return self._store.get(asset_id)

    def list_assets(self) -> Dict[str, str]:
        """Return a copy of all registered (asset_id → hash) mappings."""
        return dict(self._store)

    def remove(self, asset_id: str) -> None:
        """Remove an asset from the registry."""
        if asset_id in self._store:
            del self._store[asset_id]
            self._save()
            logger.info("Removed asset '%s' from registry.", asset_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, str]:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._store, f, indent=2, sort_keys=True)

    def _merge_master_hashes(self) -> None:
        try:
            from src.provenance.master_hashes import MASTER_HASHES
            for k, v in MASTER_HASHES.items():
                if k not in self._store:
                    self._store[k] = v
            logger.debug("Merged %d master hashes into registry.", len(MASTER_HASHES))
        except ImportError:
            logger.warning("master_hashes.py not found - skipping merge.")
        except Exception as exc:
            logger.warning("Could not merge master hashes: %s", exc)

    @staticmethod
    def _hash(path: Path, chunk_size: int = 1 << 20) -> str:
        """SHA-256 hash of a file or directory tree."""
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
                    with open(fp, "rb") as f:
                        while chunk := f.read(chunk_size):
                            h.update(chunk)
            return h.hexdigest()
        return hashlib.sha256(b"missing").hexdigest()
 