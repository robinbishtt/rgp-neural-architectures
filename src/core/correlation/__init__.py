def __getattr__(name):
    import importlib
    _LAZY = {
        'TwoPointCorrelation': ('src.core.correlation.two_point', 'TwoPointCorrelation'),
        'chi1_gauss_hermite':  ('src.core.correlation.two_point', 'chi1_gauss_hermite'),
        'critical_sigma_w2':   ('src.core.correlation.two_point', 'critical_sigma_w2'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")