def __getattr__(name):
    import importlib
    _LAZY = {
        : ('src.utils.logging.complexity_tracker', 'FisherInformationMonitor'),
        : ('src.utils.logging.complexity_tracker', 'CorrelationLengthMonitor'),
        :        ('src.utils.logging.complexity_tracker', 'ComplexityTracker'),
        :       ('src.utils.logging.complexity_tracker', 'RealTimeVisualizer'),
        :          ('src.utils.logging.complexity_tracker', 'RGFlowValidator'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.utils.logging' has no attribute {name!r}")