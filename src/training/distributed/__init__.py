def __getattr__(name):
    import importlib
    _LAZY = {
        :           ('src.training.distributed.fsdp_orchestrator', 'FSDPOrchestrator'),
        :               ('src.training.distributed.fsdp_orchestrator', 'RGFSDPConfig'),
        : ('src.training.distributed.fsdp_orchestrator', 'DistributedTrainingManager'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch and torch.distributed")
    raise AttributeError(f"module 'src.training.distributed' has no attribute {name!r}")