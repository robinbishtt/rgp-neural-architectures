def __getattr__(name):
    import importlib
    _LAZY = {
        :             ('src.core.fisher.fisher_metric',        'FisherMetric'),
        :         ('src.core.fisher.fisher_base',          'FisherMetricBase'),
        : ('src.core.fisher.eigenvalue_analyzer',  'FisherEigenvalueAnalyzer'),
        : ('src.core.fisher.analytic',             'FisherAnalyticCalculator'),
        : ('src.core.fisher.fisher_dynamic_router', 'FisherInformationEstimator'),
        :        ('src.core.fisher.fisher_dynamic_router', 'FisherDynamicRouter'),
        :      ('src.core.fisher.fisher_dynamic_router', 'FisherRegularizedLoss'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.core.fisher' has no attribute {name!r}")