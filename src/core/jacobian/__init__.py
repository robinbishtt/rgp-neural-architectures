"""src/core/jacobian - Jacobian computation. Lazy-loaded (requires torch)."""
def __getattr__(name):
    import importlib
    _LAZY = {
        'AutogradJacobian':         ('src.core.jacobian.jacobian', 'AutogradJacobian'),
        'JVPJacobian':              ('src.core.jacobian.jacobian', 'JVPJacobian'),
        'VJPJacobian':              ('src.core.jacobian.jacobian', 'VJPJacobian'),
        'FiniteDifferenceJacobian': ('src.core.jacobian.jacobian', 'FiniteDifferenceJacobian'),
        'CumulativeJacobian':       ('src.core.jacobian.jacobian', 'CumulativeJacobian'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.core.jacobian' has no attribute {name!r}")
