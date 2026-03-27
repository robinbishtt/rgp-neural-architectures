def __getattr__(name):
    import importlib
    _LAZY = {
        :         ('src.scaling.fss_analysis',      'DepthScalingFitter'),
        :  ('src.scaling.critical_exponents', 'CriticalExponentExtractor'),
        :     ('src.scaling.critical_exponents', 'CriticalExponentResult'),
        :       ('src.scaling.data_collapse',      'DataCollapseVerifier'),
        :            ('src.scaling.data_collapse',      'CollapseQuality'),
        :        ('src.scaling.bootstrap',          'BootstrapConfidence'),
        :            ('src.scaling.bootstrap',          'BootstrapResult'),
        : ('src.scaling.canonical_scaling_handler', 'CorrelationLengthEstimator'),
        :    ('src.scaling.canonical_scaling_handler', 'CanonicalScalingHandler'),
        :  ('src.scaling.canonical_scaling_handler', 'StochasticDepthController'),
        :            ('src.scaling.canonical_scaling_handler', 'AdaptiveRGDepth'),
        :             ('src.scaling.canonical_scaling_handler', 'DepthScheduler'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")