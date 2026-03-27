def __getattr__(name):
    import importlib
    _LAZY = {
        :    ('src.architectures.blocks.rgp_attention',    'ScaleInvariantAttention'),
        :       ('src.architectures.blocks.rgp_attention',    'RGMultiHeadAttention'),
        :           ('src.architectures.blocks.rgp_attention',    'RGAttentionBlock'),
        :          ('src.architectures.blocks.rgp_attention',    'RGAttentionFusion'),
        :                   ('src.architectures.blocks.rgp_moe_block',    'RGExpert'),
        :                   ('src.architectures.blocks.rgp_moe_block',    'RGRouter'),
        :                 ('src.architectures.blocks.rgp_moe_block',    'RGMoELayer'),
        :                 ('src.architectures.blocks.rgp_moe_block',    'RGMoEBlock'),
        :           ('src.architectures.blocks.rgp_moe_block',    'RGMoETransformer'),
        :           ('src.architectures.blocks.rgp_bottleneck_v2', 'InformationSieve'),
        :             ('src.architectures.blocks.rgp_bottleneck_v2', 'RGBottleneckV2'),
        :               ('src.architectures.blocks.rgp_bottleneck_v2', 'RGBlockStack'),
        :       ('src.architectures.blocks.rgp_bottleneck_v2', 'RGPBottleneckNetwork'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures.blocks' has no attribute {name!r}")