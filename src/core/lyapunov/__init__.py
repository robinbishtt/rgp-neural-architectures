"""src/core/lyapunov - Lyapunov spectrum via QR algorithms."""

# All imports are lazy to avoid loading torch when not needed.
# Direct imports below are still available via standard import syntax.

def __getattr__(name):
    import importlib
    _LAZY = {
        'ParallelQRAlgorithm': ('src.core.lyapunov.parallel_qr', 'ParallelQRAlgorithm'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
