def __getattr__(name):
    import importlib
    _LAZY = {
        :          ('src.architectures.layers.renormalized_norm', 'RenormalizedNorm'),
        :   ('src.architectures.layers.renormalized_norm', 'ScaleInvariantBatchNorm'),
        :               ('src.architectures.layers.renormalized_norm', 'RGGroupNorm'),
        :        ('src.architectures.layers.renormalized_norm', 'FisherWeightedNorm'),
        :            ('src.architectures.layers.renormalized_norm', 'MultiScaleNorm'),
        :               ('src.architectures.layers.renormalized_norm', 'RGLayerNorm'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures.layers' has no attribute {name!r}")