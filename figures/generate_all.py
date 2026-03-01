"""
figures/generate_all.py

Master figure generation script — generates all manuscript and extended
data figures in sequence. Supports fast-track mode for rapid verification.

Usage
-----
    # Full quality (requires real experiment results)
    python figures/generate_all.py --results-root results/ --output figures/out/

    # Fast-track (synthetic data, < 2 minutes on CPU)
    python figures/generate_all.py --fast-track --output figures/out/

    # Single figure
    python figures/generate_all.py --figures fig1 fig3 --fast-track
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

def _build_registry(results_root: Path, output_dir: Path, fast_track: bool) -> Dict:
    h1_results  = str(results_root / "h1")
    h2_results  = str(results_root / "h2")
    h3_results  = str(results_root / "h3")
    fss_results = str(results_root / "fss")

    return {
        "fig1": {
            "module": "figures.manuscript.generate_figure1",
            "kwargs": {
                "output_path": str(output_dir / "fig1.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Figure 1: Conceptual Framework",
        },
        "fig2": {
            "module": "figures.manuscript.generate_figure2",
            "kwargs": {
                "output_path": str(output_dir / "fig2.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Figure 2: RG Layer Mechanics",
        },
        "fig3": {
            "module": "figures.manuscript.generate_figure3",
            "kwargs": {
                "results_dir": h1_results,
                "output_path": str(output_dir / "fig3.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Figure 3: H1 Scale Correspondence",
        },
        "fig4": {
            "module": "figures.manuscript.generate_figure4",
            "kwargs": {
                "results_dir": h2_results,
                "output_path": str(output_dir / "fig4.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Figure 4: H2 Depth Scaling Law",
        },
        "fig5": {
            "module": "figures.manuscript.generate_figure5",
            "kwargs": {
                "results_dir": h3_results,
                "output_path": str(output_dir / "fig5.pdf"),
                "table_path":  str(output_dir / "table1.tex"),
                "fast_track":  fast_track,
            },
            "description": "Figure 5: H3 Architectural Advantage + Table 1",
        },
        "ed_fig1": {
            "module": "figures.extended_data.run_extended_figure1",
            "kwargs": {
                "results_dir": h1_results,
                "output_path": str(output_dir / "ed_fig1.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 1: Correlation-Length Diagnostics",
        },
        "ed_fig2": {
            "module": "figures.extended_data.run_extended_figure2",
            "kwargs": {
                "results_dir": str(results_root / "jacobian"),
                "output_path": str(output_dir / "ed_fig2.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 2: Jacobian Spectrum Evolution",
        },
        "ed_fig3": {
            "module": "figures.extended_data.run_extended_figure3",
            "kwargs": {
                "output_path": str(output_dir / "ed_fig3.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 3: Stability Phase Diagram",
        },
        "ed_fig4": {
            "module": "figures.extended_data.run_extended_figure4",
            "kwargs": {
                "results_dir": fss_results,
                "output_path": str(output_dir / "ed_fig4.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 4: FSS Data Collapse",
        },
        "ed_fig5": {
            "module": "figures.extended_data.run_extended_figure5",
            "kwargs": {
                "output_path": str(output_dir / "ed_fig5.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 5: Lyapunov Spectrum",
        },
        "ed_fig6": {
            "module": "figures.extended_data.run_extended_figure6",
            "kwargs": {
                "output_path": str(output_dir / "ed_fig6.pdf"),
                "fast_track":  fast_track,
            },
            "description": "Extended Data Figure 6: Perturbation Growth",
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_figure(key: str, spec: Dict) -> bool:
    """Import module and call generate(). Returns True on success."""
    import importlib
    print(f"\n{'─'*60}")
    print(f"  {spec['description']}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(spec["module"])
        mod.generate(**spec["kwargs"])
        elapsed = time.perf_counter() - t0
        print(f"  ✓ Done in {elapsed:.1f}s")
        return True
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def generate_all(
    figures: Optional[List[str]],
    results_root: str,
    output_dir: str,
    fast_track: bool,
    skip_on_error: bool = True,
) -> None:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry = _build_registry(Path(results_root), out, fast_track)

    to_generate = figures if figures else list(registry.keys())
    unknown = [k for k in to_generate if k not in registry]
    if unknown:
        print(f"Unknown figure keys: {unknown}")
        print(f"Valid keys: {list(registry.keys())}")
        sys.exit(1)

    header = "FAST-TRACK MODE [FAST_TRACK_UNVERIFIED]" if fast_track else "FULL QUALITY MODE"
    print(f"\n{'='*60}")
    print(f"  RGP Figure Pipeline — {header}")
    print(f"  Generating {len(to_generate)} figure(s)")
    print(f"  Output: {out.resolve()}")
    print(f"{'='*60}")

    t_total = time.perf_counter()
    passed, failed = 0, 0

    for key in to_generate:
        success = _run_figure(key, registry[key])
        if success:
            passed += 1
        else:
            failed += 1
            if not skip_on_error:
                print("\nAborting due to error (--no-skip-on-error).")
                sys.exit(1)

    elapsed_total = time.perf_counter() - t_total

    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {passed} passed, {failed} failed")
    print(f"  Total time: {elapsed_total:.1f}s")
    if fast_track:
        print("  NOTE: All outputs marked [FAST_TRACK_UNVERIFIED].")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all RGP main and extended data figures."
    )
    p.add_argument(
        "--figures", nargs="+", default=None,
        metavar="KEY",
        help="Specific figure keys to generate (e.g. fig1 fig3 ed_fig2). "
             "Default: all figures.",
    )
    p.add_argument("--results-root", default="results",
                   help="Root directory for experiment results.")
    p.add_argument("--output", default="figures/out",
                   help="Output directory for generated figures.")
    p.add_argument("--fast-track", action="store_true",
                   help="Use synthetic data (3-5 minutes, no GPU required).")
    p.add_argument("--no-skip-on-error", action="store_true",
                   help="Abort pipeline on first figure failure.")
    p.add_argument("--list", action="store_true",
                   help="List available figure keys and exit.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        registry = _build_registry(Path("results"), Path("figures/out"), False)
        print("\nAvailable figure keys:")
        for key, spec in registry.items():
            print(f"  {key:<12} {spec['description']}")
        sys.exit(0)

    generate_all(
        figures=args.figures,
        results_root=args.results_root,
        output_dir=args.output,
        fast_track=args.fast_track,
        skip_on_error=not args.no_skip_on_error,
    )
