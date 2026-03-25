from __future__ import annotations
import time, json
import numpy as np
import torch
from pathlib import Path
from src.rg_flow.operators.operators import (
    StandardRGOperator, ResidualRGOperator, WaveletRGOperator, AttentionRGOperator
)
FEATURES = 64
BATCH    = 32
def measure_operator(op: torch.nn.Module, n_iters: int = 10) -> dict:
    x = torch.randn(BATCH, FEATURES)
    n_params = sum(p.numel() for p in op.parameters())
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = op(x)
    elapsed = (time.perf_counter() - t0) / n_iters * 1000  
    x_s = x[0].detach().requires_grad_(True)
    y_s = op(x_s)
    J = torch.zeros(FEATURES, FEATURES)
    for i in range(FEATURES):
        grad = torch.autograd.grad(y_s[i], x_s, retain_graph=True)[0]
        J[i] = grad.detach()
    sv = torch.linalg.svdvals(J).detach().numpy()
    return {
        : int(n_params),
        : float(elapsed),
        : float(sv.max()),
        : float(sv.mean()),
        : int((sv > sv.max() * 0.01).sum()),
    }
def run_operator_ablation(fast_track: bool = False) -> dict:
    n_iters = 3 if fast_track else 20
    ops = {
        :  StandardRGOperator(FEATURES, FEATURES, sigma_w=1.4, sigma_b=0.3),
        :  ResidualRGOperator(FEATURES, FEATURES),
        :   WaveletRGOperator(FEATURES),
        : AttentionRGOperator(FEATURES, n_heads=4),
    }
    results = {}
    for name, op in ops.items():
        stats = measure_operator(op, n_iters=n_iters)
        results[name] = stats
        print(f"  {name:<12}: params={stats['n_params']:6d}, fwd={stats['forward_ms']:.2f}ms, "
              f"max_sv={stats['jacobian_max_sv']:.4f}, rank={stats['jacobian_rank']}")
    return results
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast-track", action="store_true")
    p.add_argument("--output", default="results/ablation/operators/")
    args = p.parse_args()
    print("=== RG Operator Ablation ===")
    results = run_operator_ablation(fast_track=args.fast_track)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "operator_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")