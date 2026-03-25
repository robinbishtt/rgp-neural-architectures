def __getattr__(name):
    import importlib
    _LAZY = {
        :     ('src.architectures.baselines.resnet_baseline', 'ResNetBaseline'),
        :   ('src.architectures.baselines.densenet_baseline', 'DenseNetBaseline'),
        :        ('src.architectures.baselines.mlp_baseline', 'MLPBaseline'),
        :        ('src.architectures.baselines.vgg_baseline', 'VGGBaseline'),
        : ('src.architectures.baselines.wavelet_baseline', 'WaveletCNNBaseline'),
        :  ('src.architectures.baselines.tensor_net_baseline', 'TensorNetBaseline'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            return getattr(importlib.import_module(mod_name), attr)
        except ImportError:
            raise ImportError(f"{name!r} requires torch")
    raise AttributeError(f"module 'src.architectures.baselines' has no attribute {name!r}")