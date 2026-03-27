def __getattr__(name):
    import importlib
    _LAZY = {
        'FisherMetric':             ('src.core.fisher.fisher_metric', 'FisherMetric'),
        'FisherMetricBase':         ('src.core.fisher.fisher_base', 'FisherMetricBase'),
        'FisherEigenvalueAnalyzer': ('src.core.fisher.eigenvalue_analyzer', 'FisherEigenvalueAnalyzer'),
        'FisherAnalyticCalculator': ('src.core.fisher.analytic', 'FisherAnalyticCalculator'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.core.fisher' has no attribute {name!r}")