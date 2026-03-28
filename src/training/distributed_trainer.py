from __future__ import annotations
import logging
import math
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
logger = logging.getLogger(__name__)
GPU_MEM_BUDGET_GB = 20.0   
LINEAR_LR_SCALE   = True   
def detect_distributed_env() -> Dict[str, int]:
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    n_gpus     = torch.cuda.device_count()
    return {
        "rank":        rank,
        "local_rank":  local_rank,
        "world_size":  world_size,
        "n_gpus":      n_gpus,
    }
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()
def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0
def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1
def is_main_rank() -> bool:
    return get_rank() == 0
def init_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> Dict[str, int]:
    env = detect_distributed_env()
    if env["world_size"] > 1:
        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available; cannot initialize distributed training."
            )
        if backend == "nccl" and env["n_gpus_node"] == 0:
            warnings.warn("No GPUs found; falling back from nccl to gloo.", stacklevel=2)
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=torch.distributed.timedelta(minutes=timeout_minutes),
        )
        if env["n_gpus_node"] > 0:
            torch.cuda.set_device(env["local_rank"])
        logger.info(
            "Distributed: rank=%d/%d  local_rank=%d  backend=%s",
            env["rank"], env["world_size"], env["local_rank"], backend,
        )
    return env
def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
@contextmanager
def distributed_context(backend: str = "nccl"):
    env = init_distributed(backend)
    try:
        yield env
    finally:
        cleanup_distributed()
def make_distributed_dataloader(
    dataset:     Dataset,
    batch_size:  int,
    rank:        int,
    world_size:  int,
    shuffle:     bool = True,
    num_workers: int  = 4,
    pin_memory:  bool = True,
    drop_last:   bool = True,
    seed:        int  = 42,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,          
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t
def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t
def wrap_ddp(
    model:                  nn.Module,
    device:                 torch.device,
    find_unused_parameters: bool = False,
    sync_bn:                bool = True,
) -> DDP:
    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )
    return ddp_model
def wrap_fsdp(
    model:           nn.Module,
    device:          torch.device,
    min_num_params:  int   = 1_000_000,
    cpu_offload:     bool  = False,
) -> nn.Module:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError:
        raise ImportError(
        )
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=min_num_params
    )
    cpu_offload_cfg = CPUOffload(offload_params=cpu_offload)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload_cfg,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
    )
    return fsdp_model
def estimate_model_gb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
    return n_params * bytes_per.get(dtype, 4) / 1e9
def measure_gradient_fisher_distributed(
    model:     nn.Module,
    X_local:   torch.Tensor,
    y_local:   torch.Tensor,
    device:    torch.device,
    criterion: Optional[nn.Module] = None,
) -> List[np.ndarray]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    unwrapped = model.module if hasattr(model, "module") else model
    layer_grads: Dict[int, torch.Tensor] = {}
    def _make_hook(k: int):
        def _hook(module, grad_in, grad_out):
            layer_grads[k] = grad_out[0].detach()
        return _hook
    handles = []
    if hasattr(unwrapped, "rg_layers"):
        for k, rg in enumerate(unwrapped.rg_layers):
            handles.append(rg.register_full_backward_hook(_make_hook(k)))
    model.zero_grad()
    out  = model(X_local.to(device))
    loss = criterion(out, y_local.to(device))
    loss.backward()
    for h in handles:
        h.remove()
    B_local   = X_local.shape[0]
    eigenvalue_lists = []
    for k in sorted(layer_grads.keys()):
        g_np = layer_grads[k].cpu().float().numpy()   
        n_k  = g_np.shape[1]
        F_local = g_np.T @ g_np   
        if is_distributed():
            F_tensor = torch.from_numpy(F_local).to(device)
            dist.all_reduce(F_tensor, op=dist.ReduceOp.SUM)
            B_total  = torch.tensor(float(B_local), device=device)
            dist.all_reduce(B_total, op=dist.ReduceOp.SUM)
            F_global = (F_tensor / B_total).cpu().numpy()
        else:
            F_global = F_local / max(B_local, 1)
        ev = np.clip(np.linalg.eigvalsh(F_global), 1e-12, None)
        eigenvalue_lists.append(np.sort(ev))
    return eigenvalue_lists
@dataclass
class DDPTrainingConfig:
    n_epochs:          int   = 100
    batch_size_per_gpu: int  = 256   
    lr:                float = 1e-3
    weight_decay:      float = 1e-4
    grad_clip_norm:    float = 1.0
    lr_scale_by_world: bool  = True  
    use_amp:           bool  = True  
    log_interval:      int   = 10    
    checkpoint_dir:    str   = "checkpoints"
    seed:              int   = 42
    fisher_batch_size: int   = 5000  
    fisher_interval:   int   = 10    
@dataclass
class DDPTrainingResult:
    train_losses:    List[float] = field(default_factory=list)
    val_losses:      List[float] = field(default_factory=list)
    val_accs:        List[float] = field(default_factory=list)
    best_val_acc:    float = 0.0
    total_epochs:    int   = 0
    world_size:      int   = 1
    effective_batch: int   = 0
    elapsed_s:       float = 0.0
class DDPTrainer:
    def __init__(
        self,
        model:     nn.Module,
        cfg:       DDPTrainingConfig,
        device:    torch.device,
        env:       Dict[str, int],
    ) -> None:
        self.cfg        = cfg
        self.device     = device
        self.env        = env
        self.rank       = env.get("rank", 0)
        self.world_size = env.get("world_size", 1)
        self.model = wrap_ddp(model, device, find_unused_parameters=False)
        effective_lr = cfg.lr * (self.world_size if cfg.lr_scale_by_world else 1)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=effective_lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.use_amp and device.type == "cuda"
        )
        self._best_val_acc = 0.0
        if is_main_rank():
            logger.info(
                "DDPTrainer: world_size=%d  lr=%.2e (effective=%.2e)  batch/gpu=%d  eff_batch=%d",
                self.world_size, cfg.lr, effective_lr,
                cfg.batch_size_per_gpu,
                cfg.batch_size_per_gpu * self.world_size,
            )
    def train(
        self,
        train_dataset: Dataset,
        val_dataset:   Dataset,
        criterion:     Optional[nn.Module] = None,
    ) -> DDPTrainingResult:
        criterion = criterion or nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.n_epochs
        )
        result = DDPTrainingResult(
            world_size=self.world_size,
            effective_batch=self.cfg.batch_size_per_gpu * self.world_size,
        )
        train_loader, train_sampler = make_distributed_dataloader(
            train_dataset,
            batch_size=self.cfg.batch_size_per_gpu,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=True,
        )
        val_loader, _ = make_distributed_dataloader(
            val_dataset,
            batch_size=self.cfg.batch_size_per_gpu * 2,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=False,
        )
        t0 = time.perf_counter()
        for epoch in range(1, self.cfg.n_epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = self._train_epoch(train_loader, criterion)
            val_loss, val_acc = self._val_epoch(val_loader, criterion)
            scheduler.step()
            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.val_accs.append(val_acc)
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                if is_main_rank():
                    self._save_checkpoint(epoch, val_acc, best=True)
            if is_main_rank() and epoch % self.cfg.log_interval == 0:
                logger.info(
                    "Epoch [%d/%d]  train=%.4f  val=%.4f  acc=%.4f  best=%.4f",
                    epoch, self.cfg.n_epochs,
                    train_loss, val_loss, val_acc, self._best_val_acc,
                )
        result.best_val_acc  = self._best_val_acc
        result.total_epochs  = self.cfg.n_epochs
        result.elapsed_s     = time.perf_counter() - t0
        return result
    def _train_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_batches  = 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp and self.device.type == "cuda"):
                out  = self.model(x)
                loss = criterion(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.detach()
            n_batches  += 1
        avg_loss = all_reduce_mean(total_loss / max(n_batches, 1))
        return float(avg_loss)
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        correct    = torch.tensor(0, device=self.device)
        total      = torch.tensor(0, device=self.device)
        for x, y in loader:
            x, y   = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            out    = self.model(x)
            total_loss += criterion(out, y).detach()
            correct    += (out.argmax(1) == y).sum()
            total      += y.size(0)
        total_loss = all_reduce_mean(total_loss / max(len(loader), 1))
        correct    = all_reduce_sum(correct.float())
        total      = all_reduce_sum(total.float())
        return float(total_loss), float(correct / total.clamp(min=1))
    def _save_checkpoint(self, epoch: int, val_acc: float, best: bool = False) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fname = "best.pt" if best else f"epoch_{epoch:04d}.pt"
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "epoch":               epoch,
            "val_acc":             val_acc,
            "model_state_dict":    unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, ckpt_dir / fname)
    def unwrapped_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model
def auto_select_trainer(
    model:       nn.Module,
    cfg:         DDPTrainingConfig,
    dtype:       torch.dtype = torch.float32,
    force_ddp:   bool = False,
    force_fsdp:  bool = False,
) -> Tuple[str, torch.device, Dict[str, int]]:
    env = detect_distributed_env()
    n_gpus = torch.cuda.device_count()
    if force_fsdp:
        strategy = "fsdp"
    elif force_ddp:
        strategy = "ddp"
    elif n_gpus == 0:
        strategy = "cpu"
    elif n_gpus == 1 and env["world_size"] == 1:
        model_gb = estimate_model_gb(model, dtype)
        free_gb  = torch.cuda.mem_get_info()[0] / 1e9
        strategy = "fsdp" if model_gb > free_gb * 0.8 else "single_gpu"
    else:
        model_gb = estimate_model_gb(model, dtype)
        gpu_gb   = torch.cuda.mem_get_info()[0] / 1e9
        strategy = "fsdp" if model_gb > gpu_gb * 0.6 else "ddp"
    if n_gpus > 0:
        device = torch.device(f"cuda:{env['local_rank']}")
    else:
        device = torch.device("cpu")
    if is_main_rank():
        model_mb = estimate_model_gb(model, dtype) * 1000
        logger.info(
            "Auto-select: n_gpus=%d  world=%d  model=%.1fMB  strategy=%s  device=%s",
            n_gpus, env["world_size"], model_mb, strategy, device,
        )
    return strategy, device, env
def train_distributed(
    model_factory:   Callable[[], nn.Module],
    train_dataset:   Dataset,
    val_dataset:     Dataset,
    cfg:             DDPTrainingConfig,
    criterion:       Optional[nn.Module] = None,
    backend:         str = "nccl",
) -> DDPTrainingResult:
    with distributed_context(backend) as env:
        n_gpus  = torch.cuda.device_count()
        device  = torch.device(f"cuda:{env['local_rank']}") if n_gpus > 0 else torch.device("cpu")
        torch.manual_seed(cfg.seed + env["rank"])
        model = model_factory()
        strategy, device, env = auto_select_trainer(model, cfg)
        if strategy in ("ddp", "single_gpu"):
            trainer = DDPTrainer(model, cfg, device, env)
        elif strategy == "fsdp":
            model   = wrap_fsdp(model, device)
            trainer = DDPTrainer(model, cfg, device, env)
            logger.info("Using FSDP sharding for large model.")
        else:
            trainer = DDPTrainer(model, cfg, device, env)
            logger.info("No GPUs found; training on CPU.")
        result = trainer.train(train_dataset, val_dataset, criterion)
    return result
def launch_command(
    script:         str,
    n_gpus:         Optional[int] = None,
    master_addr:    str  = "localhost",
    master_port:    int  = 29500,
    nnodes:         int  = 1,
    node_rank:      int  = 0,
) -> str:
    nproc = str(n_gpus) if n_gpus else "auto"
    if nnodes == 1:
        return (
            f"torchrun --standalone --nproc_per_node={nproc} {script}"
        )
    else:
        return (
            f"torchrun --nproc_per_node={nproc} --nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--master_addr={master_addr} --master_port={master_port} "
            f"{script}"
        )

class DistributedTrainer:
    """Lightweight wrapper for single-process or distributed training.

    Used primarily for testing and single-node setups where full DDP
    is not needed but distributed-training utilities are still exercised.

    Args:
        rank:       Process rank (default 0 for single-process).
        world_size: Total number of processes (default 1).
    """

    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        self.rank       = rank
        self.world_size = world_size

    def is_main_rank(self) -> bool:
        """Return True iff this is the primary (rank-0) process."""
        return self.rank == 0
