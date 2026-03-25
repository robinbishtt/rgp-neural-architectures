from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from src.core.correlation.two_point import chi1_gauss_hermite
WIDTHS    = [32, 64, 128, 256, 512, 1024]
SIGMA_W   = 1.2    
SIGMA_B   = 0.3
def finite_width_chi1_empirical(
    N:         int,
    sigma_w:   float = SIGMA_W,
    sigma_b:   float = SIGMA_B,
    n_samples: int   = 500,
    seed:      int   = 42,
) -> float:
    rng = np.random.default_rng(seed)
    chi1_samples = []
    for _ in range(n_samples):
        W   = rng.standard_normal((N, N)) * (sigma_w / np.sqrt(N))
        b   = rng.standard_normal(N) * sigma_b
        x   = rng.standard_normal(N)
        pre = W @ x + b
        h   = np.tanh(pre)
        dphi = 1.0 - h ** 2    
        jac_frob2 = np.dot(dphi ** 2, np.sum(W ** 2, axis=1))  
        chi1_samples.append(float(jac_frob2 / N))
    return float(np.mean(chi1_samples))
def run_width_ablation(fast_track: bool = False) -> dict:
    n_samples = 50 if fast_track else 500
    chi1_mf = chi1_gauss_hermite(SIGMA_W ** 2, "tanh", sigma_b2=SIGMA_B ** 2)
    eps0_mf = 1.0 - chi1_mf   
    results  = {}
    print(f"  Mean-field chi1 (N→∞): {chi1_mf:.6f}  [FIXED: includes sigma_b={SIGMA_B}]")
    print(f"  Mean-field ε₀  (N→∞): {eps0_mf:.6f}")
    print()
    for N in (WIDTHS[:3] if fast_track else WIDTHS):
        chi1_emp = finite_width_chi1_empirical(N, SIGMA_W, SIGMA_B, n_samples)
        correction    = chi1_emp - chi1_mf        
        correction_pct = correction / max(abs(chi1_mf), 1e-8) * 100
        eps0_finite    = 1.0 - chi1_emp
        mf_valid       = abs(correction) < 0.01
        results[f"N_{N}"] = {
            :                N,
            :          SIGMA_W,
            :          SIGMA_B,
            :   round(chi1_emp,  6),
            :          round(chi1_mf,   6),
            :  round(correction, 6),
            :   round(correction_pct, 3),
            :      round(eps0_finite, 6),
            :          round(eps0_mf, 6),
            : mf_valid,
            :    chi1_emp < 1.0,
        }
        mf_tag = "✓ MF valid" if mf_valid else "~ deviation > 1%"
        print(f"  N={N:5d}: chi1_emp={chi1_emp:.4f}  corr={correction:+.4f} ({correction_pct:+.2f}%)  {mf_tag}")
    if not fast_track:
        Ns    = np.array([r["N"] for r in results.values()])
        chi1s = np.array([r["chi1_empirical"] for r in results.values()])
        try:
            from scipy.optimize import curve_fit
            def _model(N, chi1_inf, c1): return chi1_inf + c1 / N
            popt, _ = curve_fit(_model, Ns, chi1s, p0=[chi1_mf, 0.1])
            results["finite_width_fit"] = {
                : round(float(popt[0]), 6),
                :  round(float(popt[1]), 6),
                :  round(chi1_mf, 6),
                :           "chi1(N) = chi1_inf + c1/N",
            }
            print(f"\n  1/N fit: chi1(N) = {popt[0]:.4f} + {popt[1]:.4f}/N")
            print(f"  (theory: {chi1_mf:.4f}; fitted: {popt[0]:.4f}; relative error: {abs(popt[0]-chi1_mf)/chi1_mf*100:.2f}%)")
        except Exception as e:
            results["finite_width_fit"] = {"error": str(e)}
    results["summary"] = {
        :       SIGMA_W,
        :       SIGMA_B,
        :       round(chi1_mf, 6),
        :         "ordered" if chi1_mf < 1.0 else "chaotic",
        :          f"FIXED: sigma_w=1.2 (chi1={chi1_mf:.3f}, k_c={-1/np.log(chi1_mf):.1f}) in ordered phase."
                         ,
    }
    return results
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output",     default="results/ablation/width/")
    args = p.parse_args()
    print("=== Width Ablation: 1/N Corrections (FIXED) ===")
    results = run_width_ablation(fast_track=args.fast_track)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "width_ablation_fixed.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}/width_ablation_fixed.json")