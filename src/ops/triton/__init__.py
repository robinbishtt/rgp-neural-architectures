def __getattr__(name):
    import importlib
    _LAZY = {
        :           ('src.ops.triton.triton_custom_kernels', 'TritonRGAttention'),
        :       ('src.ops.triton.triton_custom_kernels', 'TritonFisherEstimator'),
        :              ('src.ops.triton.triton_custom_kernels', 'TritonXiScaler'),
        :            ('src.ops.triton.triton_custom_kernels', 'OptimizedRGLayer'),
        :     ('src.ops.triton.triton_custom_kernels', 'get_triton_availability'),
        : ('src.ops.triton.triton_custom_kernels', 'benchmark_triton_vs_pytorch'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch (and optionally triton)")
    raise AttributeError(f"module 'src.ops.triton' has no attribute {name!r}")