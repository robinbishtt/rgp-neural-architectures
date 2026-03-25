from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)
class CheckpointIntegrityError(RuntimeError):
class CheckpointVerifier:
    def __init__(
        self,
        strict_hash:  bool = True,
        strict_keys:  bool = True,
        verify_rng:   bool = True,
    ) -> None:
        self.strict_hash = strict_hash
        self.strict_keys = strict_keys
        self.verify_rng  = verify_rng
    def verify(
        self,
        checkpoint_dir: Path,
        model:          Optional[nn.Module] = None,
    ) -> bool:
        checkpoint_dir = Path(checkpoint_dir)
        meta_path = checkpoint_dir / "metadata.json"
        if self.strict_hash:
            self._verify_hashes(checkpoint_dir, meta_path)
        if self.strict_keys and model is not None:
            model_pt = checkpoint_dir / "model.pt"
            if model_pt.exists():
                self._verify_model_keys(model_pt, model)
            else:
                shards = sorted(checkpoint_dir.glob("rank_*_model_shard.pt"))
                if shards:
                    self._verify_model_keys(shards[0], model)
        if self.verify_rng:
            rng_path = checkpoint_dir / "rng_state.pkl"
            if rng_path.exists():
                self._verify_rng_state(rng_path, meta_path)
        logger.info("CheckpointVerifier: %s passed all checks.", checkpoint_dir.name)
        return True
    def _verify_hashes(self, ckpt_dir: Path, meta_path: Path) -> None:
        if not meta_path.exists():
            if self.strict_hash:
                raise CheckpointIntegrityError(
                    f"metadata.json missing from {ckpt_dir}. "
                )
            logger.warning("metadata.json not found - skipping hash verification.")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        recorded_hashes: Dict[str, str] = meta.get("shard_hashes", {})
        if not recorded_hashes:
            recorded_hashes = meta.get("file_hashes", {})
        for fname, expected_hash in recorded_hashes.items():
            fpath = ckpt_dir / fname
            if not fpath.exists():
                raise CheckpointIntegrityError(
                    f"Expected checkpoint file missing: {fpath}"
                )
            actual_hash = self._sha256(fpath)
            if actual_hash != expected_hash:
                raise CheckpointIntegrityError(
                    f"SHA-256 mismatch for {fname}.\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual_hash}\n"
                )
        logger.debug("Hash verification passed for %d files.", len(recorded_hashes))
    def _verify_model_keys(self, state_file: Path, model: nn.Module) -> None:
        saved   = set(torch.load(state_file, map_location="cpu")["model_state"].keys())
        current = set(model.state_dict().keys())
        missing_in_ckpt  = current - saved
        extra_in_ckpt    = saved - current
        if missing_in_ckpt or extra_in_ckpt:
            msg = (
                f"State dict mismatch for {state_file.name}.\n"
                f"  Keys in model but not checkpoint: {missing_in_ckpt}\n"
                f"  Keys in checkpoint but not model: {extra_in_ckpt}\n"
            )
            raise CheckpointIntegrityError(msg)
        logger.debug("Model key verification passed (%d keys).", len(current))
    def _verify_rng_state(self, rng_path: Path, meta_path: Path) -> None:
        import pickle
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
        expected_sample = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            expected_sample = meta.get("rng_verification_sample")
        if expected_sample is None:
            logger.debug("No RNG verification sample recorded - skipping RNG check.")
            return
        saved_torch_state = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_state["torch"])
            sample = torch.randint(0, 1_000_000, (1,)).item()
            if sample != expected_sample:
                raise CheckpointIntegrityError(
                    f"RNG state mismatch: expected sample {expected_sample}, "
                    f"got {sample}. RNG state is corrupted."
                )
        finally:
            torch.set_rng_state(saved_torch_state)
        logger.debug("RNG state verification passed.")
    @staticmethod
    def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()