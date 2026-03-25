def __getattr__(name):
    import importlib
    _LAZY = {
        : ('src.orchestration.dag_executor', 'DAGExecutor'),
        : ('src.orchestration.dag_executor', 'Task'),
        : ('src.orchestration.pipeline', 'build_full_pipeline'),
        : ('src.orchestration.pipeline', 'build_fast_track_pipeline'),
        : ('src.orchestration.hydra_config', 'compose_config'),
        : ('src.orchestration.hydra_config', 'fast_track_overrides'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")