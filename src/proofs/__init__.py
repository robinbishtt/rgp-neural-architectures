"""src/proofs - Symbolic and numerical theorem verification."""

# All imports are lazy to avoid loading torch when not needed.
# Direct imports below are still available via standard import syntax.

def __getattr__(name):
    import importlib
    _LAZY = {
        'verify_theorem1': ('src.proofs.theorem1_fisher_transform', 'run_all_verifications'),
        'verify_theorem2': ('src.proofs.theorem2_exponential_decay', 'run_all_verifications'),
        'verify_theorem3': ('src.proofs.theorem3_depth_scaling', 'run_all_verifications'),
        'verify_critical_init': ('src.proofs.lemma_critical_init', 'run_all_verifications'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
