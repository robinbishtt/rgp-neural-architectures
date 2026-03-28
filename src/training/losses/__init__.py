def __getattr__(name):
    import importlib
    _LAZY = {
        :       ('src.training.losses.topological_loss', 'RGFlowConsistencyLoss'),
        :   ('src.training.losses.topological_loss', 'InformationBottleneckLoss'),
        : ('src.training.losses.topological_loss', 'TopologicalRegularizationLoss'),
        :              ('src.training.losses.topological_loss', 'CombinedRGLoss'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.training.losses' has no attribute {name!r}")