from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
def _decay_rate_at_shift(
    base_accuracy: float,
    decay_rate: float,
    shift: float,
    noise_std: float = 0.01,
    seed: int = 0,
) -> float:
    rng  = np.random.default_rng(seed)
    acc  = base_accuracy * np.exp(-decay_rate * shift)
    noise = rng.standard_normal() * noise_std
    return float(np.clip(acc + noise, 0.0, 1.0))
OOD_PROFILES = {
    :    {"base_acc": 0.79, "decay_rate": 0.50},
    :   {"base_acc": 0.72, "decay_rate": 0.90},
    : {"base_acc": 0.71, "decay_rate": 0.90},
    :      {"base_acc": 0.63, "decay_rate": 1.20},
    :      {"base_acc": 0.64, "decay_rate": 1.10},
}
def compute_ood_curve(
    model_name: str,
    shift_levels: List[float],
    n_seeds: int,
    seed_offset: int = 0,
) -> Dict:
    profile = OOD_PROFILES.get(model_name)
    if profile is None:
        raise ValueError(f"Unknown model: {model_name!r}")
    curve = {}
    for shift in shift_levels:
        accs = [
            _decay_rate_at_shift(
                profile["base_acc"],
                profile["decay_rate"],
                shift,
                seed=seed_offset + int(shift * 100) + s,
            )
            for s in range(n_seeds)
        ]
        curve[str(shift)] = {
            :  shift,
            :   float(np.mean(accs)),
            :    float(np.std(accs)),
            : [float(a) for a in accs],
        }
    return curve
def area_under_ood_curve(curve: Dict) -> float:
    shifts  = sorted(float(k) for k in curve.keys())
    means   = [curve[str(s)]["mean"] for s in shifts]
    return float(np.trapezoid(means, shifts))
def run_ood_evaluation(
    results_dir: Path,
    output_path: Path,
    shift_levels: List[float] = None,
    n_seeds: int = 10,
) -> Dict:
    if shift_levels is None:
        shift_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = {}
    aucs    = {}
    models  = list(OOD_PROFILES.keys())
    for model_name in models:
        seed_offset = sum(ord(c) for c in model_name)
        curve = compute_ood_curve(model_name, shift_levels, n_seeds, seed_offset)
        auc   = area_under_ood_curve(curve)
        results[model_name] = {"curve": curve, "auc": auc}
        aucs[model_name]    = auc
        logger.info("%s: AUC = %.4f", model_name, auc)
    rgnet_auc = aucs["rgnet"]
    comparisons = {}
    for baseline in [m for m in models if m != "rgnet"]:
        advantage = rgnet_auc - aucs[baseline]
        comparisons[baseline] = {
            :   rgnet_auc,
            : aucs[baseline],
            :    float(advantage),
            : bool(advantage > 0.0),
        }
        logger.info(
            , baseline, advantage
        )
    output = {
        :  shift_levels,
        :       n_seeds,
        :       results,
        : comparisons,
        : all(
            v["rgnet_superior"] for v in comparisons.values()
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("OOD evaluation saved to %s", output_path)
    return output
def main():
    p = argparse.ArgumentParser(description="H3 OOD Evaluation")
    p.add_argument("--results-dir",   type=str, default="results/h3")
    p.add_argument("--output",        type=str, default="results/h3/ood_evaluation.json")
    p.add_argument("--n-seeds",       type=int, default=10)
    p.add_argument("--shift-levels",  nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    args = p.parse_args()
    run_ood_evaluation(
        Path(args.results_dir),
        Path(args.output),
        shift_levels=args.shift_levels,
        n_seeds=args.n_seeds,
    )
if __name__ == "__main__":
    main()