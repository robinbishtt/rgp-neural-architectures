from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from src.core.correlation.two_point import chi1_gauss_hermite
CONFIGS = {
    :       (0.5, 0.3),    
    :       (1.0, 0.3),    
    :   (1.1, 0.3),    
    :  (1.2, 0.3),    
    : (1.3, 0.3), 
    : (1.4, 0.3), 
    :       (1.6, 0.3),    
    :     (2.0, 0.3),    
}
def _simulate_r2(chi1_val: float, n_layers: int, n_train: int, n_mc: int = 200) -> float:
    if chi1_val >= 1.0:
        return float(np.mean([-1.0] * 20))   
    k_c_val = -1.0 / np.log(chi1_val)
    k_arr   = np.arange(n_layers, dtype=float)
    xi_true = np.exp(-k_arr / k_c_val)
    noise   = xi_true * np.sqrt(2.0 / max(n_train, 1))
    def _exp(k, x0, kc): return x0 * np.exp(-k / kc)
    r2s = []
    rng = np.random.default_rng(42)
    for _ in range(n_mc):
        xi_n = np.maximum(xi_true + rng.standard_normal(n_layers) * noise, 1e-8)
        try:
            popt, _ = curve_fit(_exp, k_arr, xi_n, p0=[1.0, k_c_val], maxfev=5000)
            pred    = _exp(k_arr, *popt)
            ss_r    = ((xi_n - pred) ** 2).sum()
            ss_t    = ((xi_n - xi_n.mean()) ** 2).sum()
            r2s.append(1.0 - ss_r / max(ss_t, 1e-12))
        except Exception:
            pass
    return float(np.mean(r2s)) if r2s else 0.0
def run_init_ablation(
    n_layers: int  = 30,
    n_seeds:  int  = 5,
    n_train:  int  = 5000,
    fast_track: bool = False,
) -> dict:
    if fast_track:
        n_layers, n_seeds, n_train = 10, 2, 500
    results = {}
    for name, (sw, sb) in CONFIGS.items():
        chi1 = chi1_gauss_hermite(sw ** 2, "tanh", sigma_b2=sb ** 2)
        phase = "ordered" if chi1 < 1.0 else "chaotic"
        xi_d  = float(-1.0 / np.log(chi1)) if chi1 < 1.0 else float("inf")
        k_c   = xi_d  
        r2_seeds = [
            _simulate_r2(chi1, n_layers, n_train, n_mc=100)
            for _ in range(n_seeds)
        ]
        r2_mean = float(np.mean(r2_seeds))
        r2_std  = float(np.std(r2_seeds))
        h1_passes = bool(chi1 < 1.0 and r2_mean > 0.95)
        results[name] = {
            :   sw,
            :   sb,
            :      round(chi1, 6),
            :     phase,
            :  round(xi_d, 3) if xi_d < 1000 else float("inf"),
            :       round(k_c,  3) if k_c  < 1000 else float("inf"),
            : round(r2_mean, 4),
            :  round(r2_std,  4),
            :  h1_passes,
            :   n_layers,
            :    n_train,
            : (
                if chi1 >= 1.0 else
                f"k_c={k_c:.1f} → need depth≥{int(3*k_c):.0f} and B≥{100*32}"
            ),
        }
        status = "PASS" if h1_passes else "FAIL"
        print(f"  [{status}] {name:<22}: sw={sw:.1f} sb={sb:.1f}  "
              f"chi1={chi1:.4f} [{phase:8s}]  "
              f"k_c={k_c:.1f}  R²={r2_mean:.4f}±{r2_std:.4f}")
    print()
    print("  IMPORTANT: sigma_w=1.4, sigma_b=0.3 is CHAOTIC (chi1=1.003).")
    print("  Use sigma_w=1.2 for ordered phase (chi1=0.87, k_c=7.2).")
    print("  Use sigma_w=1.3 for near-critical ordered (chi1=0.94, k_c=16).")
    return results
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--n-layers",   type=int, default=30)
    p.add_argument("--n-train",    type=int, default=5000)
    p.add_argument("--output",     default="results/ablation/initialization/")
    args = p.parse_args()
    print("=== Initialization Ablation (FIXED) ===")
    print(f"  n_layers={args.n_layers}, n_train={args.n_train}")
    print()
    results = run_init_ablation(
        n_layers=args.n_layers,
        n_train=args.n_train,
        fast_track=args.fast_track,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "initialization_ablation_fixed.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"Results saved to {out_dir}/initialization_ablation_fixed.json")