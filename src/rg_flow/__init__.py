def __getattr__(name):
    import importlib
    _LAZY = {
        :    ('src.rg_flow.operators.operators',           'StandardRGOperator'),
        :    ('src.rg_flow.operators.operators',           'ResidualRGOperator'),
        :   ('src.rg_flow.operators.attention_rg_operator', 'AttentionRGOperator'),
        :     ('src.rg_flow.operators.wavelet_rg_operator', 'WaveletRGOperator'),
        :     ('src.rg_flow.operators.learned_rg_operator', 'LearnedRGOperator'),
        :          ('src.rg_flow.continuous_rg_flow', 'ODEFunction'),
        :        ('src.rg_flow.continuous_rg_flow', 'RGODEFunction'),
        :      ('src.rg_flow.continuous_rg_flow', 'NeuralODESolver'),
        :     ('src.rg_flow.continuous_rg_flow', 'ContinuousRGFlow'),
        :     ('src.rg_flow.continuous_rg_flow', 'RGNeuralODEBlock'),
        :   ('src.rg_flow.continuous_rg_flow', 'RGNeuralODENetwork'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")