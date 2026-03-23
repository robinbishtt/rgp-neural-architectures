"""
ablation/run_all_ablations.py

Master script running all ablation studies.

Usage:
    python ablation/run_all_ablations.py --fast-track   # 2-3 min
    python ablation/run_all_ablations.py                 # full run
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation.run_activation_ablation    import run_activation_ablation
from ablation.run_initialization_ablation import run_init_ablation
from ablation.run_width_ablation         import run_width_ablation
from ablation.run_depth_ablation         import run_depth_ablation
from ablation.run_operator_ablation      import run_operator_ablation
from ablation.run_skip_connection_ablation import run_skip_ablation
from ablation.run_off_critical_ablation  import run_off_critical_ablation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/")
    args = p.parse_args()

    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ablations = [
        ("Activation Functions", run_activation_ablation),
        ("Initialization Configs", run_init_ablation),
        ("Width (1/N corrections)", run_width_ablation),
        ("Depth vs L_min", run_depth_ablation),
        ("RG Operator Types", run_operator_ablation),
        ("Skip Connections", run_skip_ablation),
        ("Off-Critical Init", run_off_critical_ablation),
    ]

    all_results = {}
    for name, fn in ablations:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            result = fn(fast_track=args.fast_track)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}

    # Save consolidated results
    (out / "all_ablations.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}")
    print(f"All ablations complete in {time.time()-t0:.1f}s")
    print(f"Results saved to {out}/all_ablations.json")


if __name__ == "__main__":
    main()
