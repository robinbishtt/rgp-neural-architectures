from src.utils.seed_registry import SeedRegistry
def __getattr__(name):
    import importlib
    _LAZY = {
        'DeviceManager': ('src.utils.device_manager', 'DeviceManager'),
        'DataAuditor':   ('src.utils.provenance', 'DataAuditor'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")