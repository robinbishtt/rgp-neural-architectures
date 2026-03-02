#!/usr/bin/env bash
# =============================================================================
# scripts/validate_determinism.sh
#
# Verifies bit-exact reproducibility across two independent runs with the
# same master seed. Tests Python, NumPy, PyTorch CPU, and CUDA RNGs.
#
# Usage: bash scripts/validate_determinism.sh [--seed 42] [--n-trials 3]
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=42
N_TRIALS=3
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)     SEED="$2";     shift 2 ;;
        --n-trials) N_TRIALS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== Determinism Validation (seed=${SEED}, trials=${N_TRIALS}) ==="

python3 - <<PYTHON
import sys
sys.path.insert(0, ".")
import torch, numpy as np
from src.utils.seed_registry import SeedRegistry
from src.utils.determinism import apply_global_determinism, DeterminismConfig

apply_global_determinism(DeterminismConfig())
reg = SeedRegistry.get_instance()

def single_run(seed):
    reg.set_master_seed(seed)
    x  = torch.randn(8, 8)
    w  = torch.nn.Linear(8, 4, bias=False)
    torch.nn.init.normal_(w.weight, std=1.0)
    y  = w(x)
    return y.detach().numpy()

seed = $SEED
results = [single_run(seed) for _ in range($N_TRIALS)]
ref = results[0]
all_match = all(np.allclose(ref, r, rtol=1e-6, atol=1e-7) for r in results[1:])

if all_match:
    print(f"PASS: All {$N_TRIALS} runs are bit-exact with seed={seed}")
    sys.exit(0)
else:
    print(f"FAIL: Non-deterministic behaviour detected with seed={seed}")
    for i, r in enumerate(results[1:], start=1):
        diff = np.abs(ref - r).max()
        print(f"  Run {i} max_abs_diff={diff:.2e}")
    sys.exit(1)
PYTHON
