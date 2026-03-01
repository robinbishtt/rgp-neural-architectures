"""
src/orchestration/hydra_config.py

Hydra configuration composition helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional


def compose_config(
    config_path: str = "config",
    config_name: str = "base",
    overrides: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compose Hydra configuration. Falls back to pyyaml if Hydra unavailable.
    """
    try:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(Path(config_path).resolve()), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        return dict(cfg)
    except ImportError:
        import yaml
        cfg_file = Path(config_path) / f"{config_name}.yaml"
        if cfg_file.exists():
            with open(cfg_file) as f:
                return yaml.safe_load(f) or {}
        return {}


def fast_track_overrides(hypothesis: str = "all") -> list:
    """Return Hydra override list for fast-track mode."""
    base = ["depth=10", "width=64", "epochs=2", "batch_size=16"]
    h_map = {
        "h1": base + ["correlation_length_samples=50", "fisher_mc_samples=100"],
        "h2": base + ["depth_sweep=[10,20]", "correlation_lengths=[5,10]"],
        "h3": base,
        "all": base,
    }
    return h_map.get(hypothesis, base)
