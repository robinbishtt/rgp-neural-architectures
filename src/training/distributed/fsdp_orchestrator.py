from __future__ import annotations
import os
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)
from torch.distributed.fsdp.api import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
class FSDPOrchestrator:
    def __init(
        self,
        world_size: int = 8,
        rank: Optional[int] = None,
        sharding_strategy: str = 'FULL_SHARD',
        mixed_precision: str = 'bf16',
        backward_prefetch: str = 'BACKWARD_PRE',
        cpu_offload: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = True,
    ) -> None:
        self.world_size = world_size
        self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.sharding_strategy = self._parse_sharding_strategy(sharding_strategy)
        self.mixed_precision = self._parse_mixed_precision(mixed_precision)
        self.backward_prefetch = self._parse_backward_prefetch(backward_prefetch)
        self.cpu_offload = CPUOffload(offload_params=cpu_offload)
        self.limit_all_gathers = limit_all_gathers
        self.use_orig_params = use_orig_params
        self.fsdp_wrapped_model: Optional[FSDP] = None
        self._init_distributed()
    def _init_distributed(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank,
            )
        torch.cuda.set_device(self.local_rank)
    def _parse_sharding_strategy(self, strategy: str) -> ShardingStrategy:
        strategies = {
            : ShardingStrategy.FULL_SHARD,
            : ShardingStrategy.SHARD_GRAD_OP,
            : ShardingStrategy.NO_SHARD,
            : ShardingStrategy.HYBRID_SHARD,
        }
        return strategies.get(strategy, ShardingStrategy.FULL_SHARD)
    def _parse_mixed_precision(self, precision: str) -> MixedPrecision:
        precisions = {
            : MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            : MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            : MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
        }
        return precisions.get(precision, precisions['bf16'])
    def _parse_backward_prefetch(self, prefetch: str) -> BackwardPrefetch:
        prefetches = {
            : BackwardPrefetch.BACKWARD_PRE,
            : BackwardPrefetch.BACKWARD_POST,
        }
        return prefetches.get(prefetch, BackwardPrefetch.BACKWARD_PRE)
    def wrap_model(
        self,
        model: nn.Module,
        auto_wrap_policy: Optional[Callable] = None,
        layer_cls: Optional[type] = None,
    ) -> FSDP:
        if auto_wrap_policy is None and layer_cls is not None:
            auto_wrap_policy = ModuleWrapPolicy({layer_cls})
        device_id = torch.cuda.current_device()
        self.fsdp_wrapped_model = FSDP(
            model,
            device_id=device_id,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self.mixed_precision,
            backward_prefetch=self.backward_prefetch,
            cpu_offload=self.cpu_offload,
            limit_all_gathers=self.limit_all_gathers,
            use_orig_params=self.use_orig_params,
        )
        return self.fsdp_wrapped_model
    def apply_activation_checkpointing(
        self,
        model: nn.Module,
        checkpoint_layer_cls: type,
    ) -> None:
        check_fn = lambda submodule: isinstance(submodule, checkpoint_layer_cls)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=check_fn,
        )
    def get_model_sharding_info(self) -> Dict[str, Any]:
        if self.fsdp_wrapped_model is None:
            return {'error': 'Model not wrapped yet'}
        info = {
            : str(self.sharding_strategy),
            : str(self.mixed_precision.param_dtype),
            : self.world_size,
            : self.rank,
            : self.local_rank,
        }
        total_params = sum(p.numel() for p in self.fsdp_wrapped_model.parameters())
        info['total_parameters'] = total_params
        local_params = sum(
            p.numel() for p in self.fsdp_wrapped_model.parameters()
            if p.device == torch.device(f'cuda:{self.local_rank}')
        )
        info['local_parameters'] = local_params
        info['sharding_ratio'] = local_params / max(total_params, 1)
        return info
    def all_reduce_gradients(self) -> None:
        if self.fsdp_wrapped_model is not None:
            return
        for param in self.fsdp_wrapped_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVERAGE)
    def synchronize(self) -> None:
        if dist.is_initialized():
            dist.barrier()
    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
class RGFSDPConfig:
    def __init(
        self,
        model_size: str = '400M',
        num_gpus: int = 8,
        enable_mixed_precision: bool = True,
        precision: str = 'bf16',
        enable_activation_checkpointing: bool = True,
        enable_cpu_offload: bool = False,
        sharding_strategy: str = 'FULL_SHARD',
    ) -> None:
        self.model_size = model_size
        self.num_gpus = num_gpus
        self.enable_mixed_precision = enable_mixed_precision
        self.precision = precision
        self.enable_activation_checkpointing = enable_activation_checkpointing
        self.enable_cpu_offload = enable_cpu_offload
        self.sharding_strategy = sharding_strategy
        self._setup_config()
    def _setup_config(self) -> None:
        configs = {
            : {
                : 1e6,
                : 1,
            },
            : {
                : 5e6,
                : 1,
            },
            : {
                : 50e6,
                : 2,
            },
            : {
                : 100e6,
                : 4,
            },
        }
        config = configs.get(self.model_size, configs['400M'])
        self.min_num_params = config['min_num_params']
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
    def create_orchestrator(self) -> FSDPOrchestrator:
        return FSDPOrchestrator(
            world_size=self.num_gpus,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self.precision if self.enable_mixed_precision else 'fp32',
            cpu_offload=self.enable_cpu_offload,
        )
    def get_auto_wrap_policy(self) -> Callable:
        return size_based_auto_wrap_policy(
            min_num_params=self.min_num_params,
        )
    def to_dict(self) -> Dict[str, Any]:
        return {
            : self.model_size,
            : self.num_gpus,
            : self.precision if self.enable_mixed_precision else 'fp32',
            : self.enable_activation_checkpointing,
            : self.enable_cpu_offload,
            : self.sharding_strategy,
            : self.gradient_accumulation_steps,
        }
class DistributedTrainingManager:
    def __init(
        self,
        orchestrator: FSDPOrchestrator,
        config: RGFSDPConfig,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config
        self.model: Optional[FSDP] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.global_step = 0
        self.epoch = 0
    def setup_model(
        self,
        model: nn.Module,
        layer_cls: Optional[type] = None,
    ) -> FSDP:
        auto_wrap_policy = self.config.get_auto_wrap_policy()
        self.model = self.orchestrator.wrap_model(
            model,
            auto_wrap_policy=auto_wrap_policy,
            layer_cls=layer_cls,
        )
        if self.config.enable_activation_checkpointing and layer_cls is not None:
            self.orchestrator.apply_activation_checkpointing(
                self.model,
                checkpoint_layer_cls=layer_cls,
            )
        return self.model
    def setup_optimizer(
        self,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        **optimizer_kwargs,
    ) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError('Model must be set up before optimizer')
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )
        return self.optimizer
    def setup_scheduler(
        self,
        scheduler_cls: type,
        **scheduler_kwargs,
    ) -> Any:
        if self.optimizer is None:
            raise RuntimeError('Optimizer must be set up before scheduler')
        self.scheduler = scheduler_cls(
            self.optimizer,
            **scheduler_kwargs,
        )
        return self.scheduler
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError('Model and optimizer must be set up before training')
        self.model.train()
        inputs, targets = batch
        inputs = inputs.cuda(self.orchestrator.local_rank)
        targets = targets.cuda(self.orchestrator.local_rank)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {
            : loss.item(),
            : self.optimizer.param_groups[0]['lr'],
            : self.global_step,
        }
    def save_checkpoint(
        self,
        path: str,
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.model is None:
            raise RuntimeError('Model must be set up before saving checkpoint')
        state = {
            : self.model.state_dict(),
            : self.optimizer.state_dict() if self.optimizer else None,
            : self.scheduler.state_dict() if self.scheduler else None,
            : self.global_step,
            : self.epoch,
        }
        if additional_state is not None:
            state.update(additional_state)
        if self.orchestrator.rank == 0:
            torch.save(state, path)
        self.orchestrator.synchronize()
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        state = torch.load(path, map_location=f'cuda:{self.orchestrator.local_rank}')
        if self.model is not None:
            self.model.load_state_dict(state['model_state_dict'])
        if self.optimizer is not None and state.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if self.scheduler is not None and state.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.global_step = state.get('global_step', 0)
        self.epoch = state.get('epoch', 0)
        return state
    def get_training_stats(self) -> Dict[str, Any]:
        stats = {
            : self.global_step,
            : self.epoch,
            : self.orchestrator.world_size,
            : self.orchestrator.rank,
        }
        sharding_info = self.orchestrator.get_model_sharding_info()
        stats.update(sharding_info)
        return stats