def __getattr__(name):
    import importlib
    _LAZY = {
        'LevelSpacingDistribution':  ('src.core.spectral.level_spacing', 'LevelSpacingDistribution'),
        'empirical_spectral_density': ('src.core.spectral.empirical_density', 'empirical_spectral_density'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")