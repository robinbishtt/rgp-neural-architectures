def __getattr__(name):
    import importlib
    _LAZY = {
        :    ('src.architectures.equivariant.symmetry_equivariance_engine', 'RotationEquivariantConv'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine', 'TranslationEquivariantPool'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine', 'ScaleEquivariantProjection'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine', 'SymmetryEquivariantRGBlock'),
        :  ('src.architectures.equivariant.symmetry_equivariance_engine', 'GroupEquivariantAttention'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine', 'SymmetryEquivarianceEngine'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures.equivariant' has no attribute {name!r}")