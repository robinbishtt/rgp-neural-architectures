"""src/provenance - SHA-256 data provenance and checksum verification."""

# All imports are lazy to avoid loading torch when not needed.
# Direct imports below are still available via standard import syntax.

def __getattr__(name):
    import importlib
    _LAZY = {
        'DataAuditor': ('src.provenance.data_auditor', 'DataAuditor'),
        'MASTER_HASHES': ('src.provenance.master_hashes', 'MASTER_HASHES'),
        'get_expected_hash': ('src.provenance.master_hashes', 'get_expected_hash'),
        'is_registered': ('src.provenance.master_hashes', 'is_registered'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
