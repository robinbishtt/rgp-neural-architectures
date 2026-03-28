def __getattr__(name):
    import importlib
    _LAZY = {
        'run_theorem1_verifications': ('src.proofs.theorem1_fisher_transform', 'run_all_verifications'),
        'run_theorem2_verifications': ('src.proofs.theorem2_exponential_decay', 'run_all_verifications'),
        'run_theorem3_verifications': ('src.proofs.theorem3_depth_scaling', 'run_all_verifications'),
        'run_lemma_verifications':    ('src.proofs.lemma_critical_init', 'run_all_verifications'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")