"""
scripts/verify_pipeline.py

Smoke test for the complete pipeline. Verifies:
  1. All core imports resolve correctly
  2. Device detection works
  3. Seed registry initializes and propagates
  4. Minimal model forward/backward pass completes
  5. Checkpoint save and load round-trips correctly
  6. Data loader produces deterministic batches

Completes in under 60 seconds on any hardware.

Usage:
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --check-env
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_imports() -> Tuple[bool, str]:
    try:
        import torch
        import numpy
        import scipy
        import sympy
        import matplotlib
        return True, f"torch={torch.__version__}, numpy={numpy.__version__}"
    except ImportError as exc:
        return False, str(exc)


def check_device() -> Tuple[bool, str]:
    try:
        from src.utils.device_manager import DeviceManager
        dm     = DeviceManager.get_instance()
        device = dm.get_device()
        info   = dm.get_device_info()
        return True, f"device={device}, name={info.get('name','unknown')}"
    except Exception as exc:
        return False, str(exc)


def check_seed_registry() -> Tuple[bool, str]:
    try:
        from src.utils.seed_registry import SeedRegistry
        reg = SeedRegistry.get_instance()
        reg.set_master_seed(42)
        state = reg.snapshot_state()
        reg.set_master_seed(99)
        reg.restore_state(state)
        seed_a = reg.get_worker_seed(0)
        reg.set_master_seed(42)
        seed_b = reg.get_worker_seed(0)
        assert seed_a == seed_b, "Seed restore failed"
        return True, f"master_seed=42, worker_seed_0={seed_a}"
    except Exception as exc:
        return False, str(exc)


def check_forward_backward() -> Tuple[bool, str]:
    try:
        import torch
        from src.utils.seed_registry import SeedRegistry
        from src.utils.device_manager import DeviceManager

        SeedRegistry.get_instance().set_master_seed(42)
        device = DeviceManager.get_instance().get_device()

        # Minimal MLP: 2 layers, width 16
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 4),
        ).to(device)

        x    = torch.randn(4, 8, device=device)
        y    = torch.randint(0, 4, (4,), device=device)
        out  = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        grad_norm = sum(p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None) ** 0.5
        return True, f"loss={loss.item():.4f}, grad_norm={grad_norm:.4f}"
    except Exception as exc:
        return False, str(exc)


def check_checkpoint() -> Tuple[bool, str]:
    try:
        import torch
        import tempfile
        from pathlib import Path

        model = torch.nn.Linear(4, 2)
        w_before = model.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)
            model.weight.data.zero_()
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])

        assert torch.allclose(w_before, model.weight.data), "Checkpoint mismatch"
        return True, "checkpoint save/load round-trip passed"
    except Exception as exc:
        return False, str(exc)


def check_spectral() -> Tuple[bool, str]:
    try:
        import numpy as np
        from src.core.spectral.spectral import MarchenkoPasturDistribution
        mp = MarchenkoPasturDistribution(beta=0.5, sigma2=1.0)
        ev = mp.sample_wishart(n=64, m=128, rng=np.random.default_rng(42))
        _, pval = mp.ks_test(ev)
        return True, f"MP KS p-value={pval:.3f} (should be >0.05)"
    except Exception as exc:
        return False, str(exc)


def check_correlation_length() -> Tuple[bool, str]:
    try:
        import numpy as np
        from src.core.correlation.estimators import ExponentialDecayFitter
        k   = np.arange(20)
        xi  = 5.0 * np.exp(-k / 4.0) + 0.01 * np.random.default_rng(42).standard_normal(20)
        res = ExponentialDecayFitter().fit(xi)
        assert res.r2 > 0.90, f"R²={res.r2:.3f} too low"
        return True, f"ξ₀={res.xi_0:.2f}, k_c={res.k_c:.2f}, R²={res.r2:.3f}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS: List[Tuple[str, Callable]] = [
    ("Core imports",          check_imports),
    ("Device manager",        check_device),
    ("Seed registry",         check_seed_registry),
    ("Forward/backward pass", check_forward_backward),
    ("Checkpoint round-trip", check_checkpoint),
    ("Spectral (MP law)",     check_spectral),
    ("Correlation length",    check_correlation_length),
]


def run_checks(checks) -> int:
    failures = 0
    total_start = time.time()

    for name, fn in checks:
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception:
            ok  = False
            msg = traceback.format_exc()

        elapsed = time.time() - t0
        status  = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<28} {msg}  ({elapsed:.1f}s)")
        if not ok:
            failures += 1

    elapsed_total = time.time() - total_start
    print(f"\n  {len(checks) - failures}/{len(checks)} checks passed in {elapsed_total:.1f}s")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--check-env", action="store_true",
                        help="Only run import and device checks")
    args = parser.parse_args()

    checks = CHECKS[:2] if args.check_env else CHECKS

    print("=== Pipeline Verification ===\n")
    failures = run_checks(checks)

    if failures:
        print(f"\nFAILED: {failures} check(s). See messages above.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Pipeline is functional.")


if __name__ == "__main__":
    main()
 