from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
logger = logging.getLogger(__name__)
class DistributedCheckpoint:
    def __init__(
        self,
        checkpoint_dir:     str   = "checkpoints",
        save_every_n_steps: int   = 1000,
        keep_last_n:        int   = 3,
    ) -> None:
        self.checkpoint_dir     = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n        = keep_last_n
        self._saved: List[Path] = []
        if not dist.is_available() or not dist.is_initialized():
            logger.warning(
            )
            self._rank = 0
            self._world_size = 1
        else:
            self._rank       = dist.get_rank()
            self._world_size = dist.get_world_size()
    def save(
        self,
        step:      int,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics:   Optional[Dict[str, Any]] = None,
        is_best:   bool = False,
    ) -> Optional[Path]:
        ckpt_dir = self.checkpoint_dir / f"step_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        shard_path = ckpt_dir / f"rank_{self._rank:04d}_model_shard.pt"
        torch.save(
            {
                :     unwrapped.state_dict(),
                : optimizer.state_dict() if optimizer else None,
                :            step,
                :            self._rank,
                :      self._world_size,
            },
            shard_path,
        )
        logger.debug("Rank %d saved shard → %s", self._rank, shard_path)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self._rank == 0:
            self._write_metadata(ckpt_dir, step, metrics or {})
            self._rotate_checkpoints(ckpt_dir)
            if is_best:
                best_dir = self.checkpoint_dir / "best"
                if best_dir.exists() or best_dir.is_symlink():
                    if best_dir.is_symlink():
                        best_dir.unlink()
                    else:
                        import shutil
                        shutil.rmtree(best_dir)
                best_dir.symlink_to(ckpt_dir.resolve())
            return ckpt_dir
        return None
    def load(
        self,
        checkpoint_dir: Optional[Path] = None,
        model:          Optional[nn.Module] = None,
        optimizer:      Optional[torch.optim.Optimizer] = None,
        map_location:   Optional[Any] = None,
    ) -> Dict[str, Any]:
        if checkpoint_dir is None:
            checkpoint_dir = self._find_latest()
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoint_dir}"
            )
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        shard_files = sorted(checkpoint_dir.glob("rank_*_model_shard.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No shards in {checkpoint_dir}")
        target = checkpoint_dir / f"rank_{rank:04d}_model_shard.pt"
        if not target.exists():
            target = shard_files[0]
            logger.warning("Rank %d shard not found; loading %s instead.", rank, target)
        state = torch.load(target, map_location=map_location or "cpu")
        if model is not None:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.load_state_dict(state["model_state"])
        if optimizer is not None and state.get("optimizer_state") is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        meta_path = checkpoint_dir / "metadata.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        logger.info("Loaded checkpoint from %s (step %d)", checkpoint_dir, state.get("step", -1))
        return {"step": state.get("step", 0), "metrics": meta.get("metrics", {})}
    def _write_metadata(self, ckpt_dir: Path, step: int, metrics: Dict) -> None:
        shard_files = sorted(ckpt_dir.glob("rank_*_model_shard.pt"))
        import hashlib
        hashes = {}
        for sf in shard_files:
            h = hashlib.sha256()
            with open(sf, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            hashes[sf.name] = h.hexdigest()
        meta = dict(step=step, world_size=self._world_size,
                    shard_hashes=hashes, metrics=metrics)
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    def _rotate_checkpoints(self, new_dir: Path) -> None:
        self._saved.append(new_dir)
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                import shutil
                shutil.rmtree(old)
                logger.debug("Rotated old checkpoint %s", old)
    def _find_latest(self) -> Optional[Path]:
        dirs = sorted(self.checkpoint_dir.glob("step_*"))
        return dirs[-1] if dirs else None