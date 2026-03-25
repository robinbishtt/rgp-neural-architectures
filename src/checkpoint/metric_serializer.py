from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
class MetricStateSerializer:
    def save(self, metrics: Dict[str, Any], path: Path) -> None:
        out = {}
        for k, v in metrics.items():
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        with open(Path(path) / "metrics.json", "w") as f:
            json.dump(out, f, indent=2)
    def load(self, path: Path) -> Dict[str, Any]:
        metrics_file = Path(path) / "metrics.json"
        if not metrics_file.exists():
            return {}
        with open(metrics_file) as f:
            return json.load(f)