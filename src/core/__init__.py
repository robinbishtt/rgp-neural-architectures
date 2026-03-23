"""src/core - Mathematical foundations. All torch imports are lazy."""
from src.core.rg_flow_solver import RGFlowSolver, BetaFunctionSolver
from src.core.correlation.two_point import chi1_gauss_hermite, critical_sigma_w2

def __getattr__(name):
    import importlib
    _LAZY = {
        'FisherMetric':          ('src.core.fisher.fisher_metric', 'FisherMetric'),
        'FisherEigenvalueAnalyzer': ('src.core.fisher.eigenvalue_analyzer', 'FisherEigenvalueAnalyzer'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.core' has no attribute {name!r}")
