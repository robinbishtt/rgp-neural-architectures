"""src/architectures/rg_net - RG-Net architectures. Lazy-loaded."""
def __getattr__(name):
    import importlib
    _LAZY = {
        'RGLayer':          ('src.architectures.rg_net.rg_net', 'RGLayer'),
        'RGNetStandard':    ('src.architectures.rg_net.rg_net', 'RGNetStandard'),
        'RGNet':            ('src.architectures.rg_net.rg_net', 'RGNetStandard'),
        'build_rg_net':     ('src.architectures.rg_net.rg_net', 'build_rg_net'),
        'RGNetFactory':     ('src.architectures.rg_net.rg_net_factory', 'RGNetFactory'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures.rg_net' has no attribute {name!r}")
