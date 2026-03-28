def __getattr__(name):
    import importlib
    _LAZY = {
        :       ('src.architectures.rg_net.rg_net', 'RGLayer'),
        : ('src.architectures.rg_net.rg_net', 'RGNetStandard'),
        :  ('src.architectures.rg_net.rg_net_factory', 'RGNetFactory'),
        : ('src.architectures.blocks.rgp_attention',    'ScaleInvariantAttention'),
        :        ('src.architectures.blocks.rgp_attention',    'RGAttentionBlock'),
        :       ('src.architectures.blocks.rgp_attention',    'RGAttentionFusion'),
        :              ('src.architectures.blocks.rgp_moe_block',    'RGMoEBlock'),
        :        ('src.architectures.blocks.rgp_moe_block',    'RGMoETransformer'),
        :          ('src.architectures.blocks.rgp_bottleneck_v2','RGBottleneckV2'),
        :        ('src.architectures.layers.renormalized_norm','RenormalizedNorm'),
        :             ('src.architectures.layers.renormalized_norm','RGLayerNorm'),
        :          ('src.architectures.layers.renormalized_norm','MultiScaleNorm'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine','SymmetryEquivarianceEngine'),
        : ('src.architectures.equivariant.symmetry_equivariance_engine','SymmetryEquivariantRGBlock'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures' has no attribute {name!r}")