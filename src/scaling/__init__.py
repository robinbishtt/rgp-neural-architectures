def __getattr__(name):
    import importlib
    _LAZY = {
        'DepthScalingFitter':       ('src.scaling.fss_analysis', 'DepthScalingFitter'),
        'CriticalExponentExtractor': ('src.scaling.critical_exponents', 'CriticalExponentExtractor'),
        'CriticalExponentResult':   ('src.scaling.critical_exponents', 'CriticalExponentResult'),
        'DataCollapseVerifier':     ('src.scaling.data_collapse', 'DataCollapseVerifier'),
        'CollapseQuality':          ('src.scaling.data_collapse', 'CollapseQuality'),
        'BootstrapConfidence':      ('src.scaling.bootstrap', 'BootstrapConfidence'),
        'BootstrapResult':          ('src.scaling.bootstrap', 'BootstrapResult'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")