def __getattr__(name):
    import importlib
    _LAZY = {
        : ('src.telemetry.telemetry_logger', 'TelemetryLogger'),
        :     ('src.telemetry.hdf5_storage',     'HDF5Storage'),
        :    ('src.telemetry.jsonl_storage',    'JSONLStorage'),
        :  ('src.telemetry.parquet_storage',  'ParquetStorage'),
    }
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            raise ImportError(f"Cannot import {name!r} - torch may be required")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")