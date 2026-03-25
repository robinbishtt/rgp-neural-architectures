from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parents[2]
_OUT  = _ROOT / "results" / "supplementary"
sys.path.insert(0, str(_ROOT))
_BUDGET = {
    : {
        :   0.15,
        : 0.22,
        :     0.08,
        :  0.31,
        :  0.01,
        :         "layers",
    },
    : {
        :   0.003,
        : 0.004,
        :     0.0,
        :  0.001,
        :  0.001,
        :         "dimensionless",
    },
    : {
        :   0.05,
        : 0.08,
        :     0.02,
        :  0.04,
        :  0.01,
        :         "dimensionless",
    },
    : {
        :   0.3,
        : 0.5,
        :     0.0,
        :  0.1,
        :  0.05,
        :         "%",
    },
    : {
        :   0.010,
        : 0.015,
        :     0.005,
        :  0.008,
        :  0.001,
        :         "dimensionless",
    },
}
def _total_uncertainty(budget: dict) -> float:
    keys = ["statistical", "seed_variation", "mc_fisher", "finite_width", "numerical_fp"]
    return float(np.sqrt(sum(budget[k] ** 2 for k in keys)))
def run(fast_track: bool = False) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for obs, budget in _BUDGET.items():
        total = _total_uncertainty(budget)
        rows.append({"observable": obs, **budget, "total": round(total, 4)})
    with open(_OUT / "table_S3.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    header = "observable,statistical,seed_variation,mc_fisher,finite_width,numerical_fp,total,units"
    csv_lines = [header] + [
        f"{r['observable']},{r['statistical']},{r['seed_variation']},"
        f"{r['mc_fisher']},{r['finite_width']},{r['numerical_fp']},{r['total']},{r['units']}"
        for r in rows
    ]
    with open(_OUT / "table_S3.csv", "w") as f:
        f.write("\n".join(csv_lines))
    md = [
        ,
        ,
        ,
        ,
        ,
        ,
    ] + [
        f"| {r['observable']} | {r['statistical']} | {r['seed_variation']} "
        f"| {r['mc_fisher']} | {r['finite_width']} | {r['numerical_fp']} "
        f"| **{r['total']}** | {r['units']} |"
        for r in rows
    ]
    with open(_OUT / "table_S3.md", "w") as f:
        f.write("\n".join(md))
    print(f"Table S3 written to {_OUT}")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-track", action="store_true")
    args = parser.parse_args()
    run(fast_track=args.fast_track)
if __name__ == "__main__":
    main()