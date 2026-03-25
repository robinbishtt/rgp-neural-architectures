from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional
logger = logging.getLogger(__name__)
_DEFAULT_REGISTRY_PATH = Path("results") / "provenance" / "checksum_registry.json"
class ChecksumRegistry:
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
    def register(
        self,
        asset_id:  str,
        path:      Path,
        overwrite: bool = False,
    ) -> str:
        if asset_id in self._store and not overwrite:
            raise ValueError(
                f"Asset '{asset_id}' already registered.  "
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
            )
        logger.debug("Verified '%s' ✓", asset_id)
        return True
    def get(self, asset_id: str) -> Optional[str]:
        return self._store.get(asset_id)
    def list_assets(self) -> Dict[str, str]:
        return dict(self._store)
    def remove(self, asset_id: str) -> None:
        if asset_id in self._store:
            del self._store[asset_id]
            self._save()
            logger.info("Removed asset '%s' from registry.", asset_id)
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