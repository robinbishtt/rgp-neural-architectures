"""
figures/generate_all.py

Master figure generation script — generates all manuscript figures, extended
data figures and tables, and supplementary figures and tables in sequence.
Supports fast-track mode for rapid verification without real experiment results.

Usage
-----
    # Full quality (requires real experiment results in results/)
    python figures/generate_all.py --results-root results/ --output figures/out/

    # Fast-track (synthetic data, < 5 minutes on CPU)
    python figures/generate_all.py --fast-track --output figures/out/

    # Single figure or table
    python figures/generate_all.py --figures fig1 fig3 --fast-track

    # Only supplementary group
    python figures/generate_all.py --group supplementary --fast-track

    # List all available keys
    python figures/generate_all.py --list
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Figure and table registry
# ---------------------------------------------------------------------------

def _build_registry(results_root: Path, output_dir: Path, fast_track: bool) -> Dict:
    r    = results_root
    o    = output_dir
    ft   = fast_track

    registry = {
        # ------------------------------------------------------------------
        # Manuscript Figures 1–5
        # ------------------------------------------------------------------
        "fig1": {
            "module": "figures.manuscript.generate_figure1",
            "fn":     "generate",
            "kwargs": {"output_path": str(o / "fig1.pdf"), "fast_track": ft},
            "group":  "manuscript",
            "description": "Figure 1: Conceptual Framework",
        },
        "fig2": {
            "module": "figures.manuscript.generate_figure2",
            "fn":     "generate",
            "kwargs": {"output_path": str(o / "fig2.pdf"), "fast_track": ft},
            "group":  "manuscript",
            "description": "Figure 2: RG Layer Mechanics",
        },
        "fig3": {
            "module": "figures.manuscript.generate_figure3",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "h1"), "output_path": str(o / "fig3.pdf"),
                       "fast_track": ft},
            "group":  "manuscript",
            "description": "Figure 3: H1 Scale Correspondence",
        },
        "fig4": {
            "module": "figures.manuscript.generate_figure4",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "h2"), "output_path": str(o / "fig4.pdf"),
                       "fast_track": ft},
            "group":  "manuscript",
            "description": "Figure 4: H2 Depth Scaling Law",
        },
        "fig5": {
            "module": "figures.manuscript.generate_figure5",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "h3"), "output_path": str(o / "fig5.pdf"),
                       "table_path": str(o / "table1.tex"), "fast_track": ft},
            "group":  "manuscript",
            "description": "Figure 5: H3 Architectural Advantage + Table 1",
        },
        # ------------------------------------------------------------------
        # Extended Data Figures 1–11
        # ------------------------------------------------------------------
        "ed_fig1": {
            "module": "figures.extended_data.run_extended_figure1",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "h1"), "output_path": str(o / "ed_fig1.pdf"),
                       "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 1: Correlation-Length Diagnostics",
        },
        "ed_fig2": {
            "module": "figures.extended_data.run_extended_figure2",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "jacobian"), "output_path": str(o / "ed_fig2.pdf"),
                       "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 2: Jacobian Spectrum Evolution",
        },
        "ed_fig3": {
            "module": "figures.extended_data.run_extended_figure3",
            "fn":     "generate",
            "kwargs": {"output_path": str(o / "ed_fig3.pdf"), "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 3: Stability Phase Diagram",
        },
        "ed_fig4": {
            "module": "figures.extended_data.run_extended_figure4",
            "fn":     "generate",
            "kwargs": {"results_dir": str(r / "fss"), "output_path": str(o / "ed_fig4.pdf"),
                       "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 4: FSS Data Collapse",
        },
        "ed_fig5": {
            "module": "figures.extended_data.run_extended_figure5",
            "fn":     "generate",
            "kwargs": {"output_path": str(o / "ed_fig5.pdf"), "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 5: Lyapunov Spectrum",
        },
        "ed_fig6": {
            "module": "figures.extended_data.run_extended_figure6",
            "fn":     "generate",
            "kwargs": {"output_path": str(o / "ed_fig6.pdf"), "fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Figure 6: Perturbation Growth",
        },
        "ed_fig7": {
            "module": "figures.extended_data.run_extended_figure7",
            "fn":     "main",
            "kwargs": {"fast_track": ft, "output": Path(o / "ed_fig7.pdf")},
            "group":  "extended_data",
            "description": "Extended Data Figure 7: Lyapunov Spectrum Depth Dependence",
        },
        "ed_fig8": {
            "module": "figures.extended_data.run_extended_figure8",
            "fn":     "main",
            "kwargs": {"fast_track": ft, "output": Path(o / "ed_fig8.pdf")},
            "group":  "extended_data",
            "description": "Extended Data Figure 8: RMT Validation",
        },
        "ed_fig9": {
            "module": "figures.extended_data.run_extended_figure9",
            "fn":     "main",
            "kwargs": {"fast_track": ft, "output": Path(o / "ed_fig9.pdf")},
            "group":  "extended_data",
            "description": "Extended Data Figure 9: FSS Collapse Quality",
        },
        "ed_fig10": {
            "module": "figures.extended_data.run_extended_figure10",
            "fn":     "main",
            "kwargs": {"fast_track": ft, "output": Path(o / "ed_fig10.pdf")},
            "group":  "extended_data",
            "description": "Extended Data Figure 10: RG Operator Ablation",
        },
        "ed_fig11": {
            "module": "figures.extended_data.run_extended_figure11",
            "fn":     "main",
            "kwargs": {"fast_track": ft, "output": Path(o / "ed_fig11.pdf")},
            "group":  "extended_data",
            "description": "Extended Data Figure 11: Continuous OOD Shift",
        },
        # ------------------------------------------------------------------
        # Extended Data Tables 1–3
        # ------------------------------------------------------------------
        "ed_table1": {
            "module": "figures.extended_data.generate_extended_table1",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Table 1: Spectral Statistics and RG Diagnostics",
        },
        "ed_table2": {
            "module": "figures.extended_data.generate_extended_table2",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Table 2: Hyperparameter Sweep Results",
        },
        "ed_table3": {
            "module": "figures.extended_data.generate_extended_table3",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "extended_data",
            "description": "Extended Data Table 3: Convergence Metrics and Numerical Stability",
        },
        # ------------------------------------------------------------------
        # Supplementary Figures S1–S4
        # ------------------------------------------------------------------
        "sup_fig1": {
            "module": "figures.supplementary.generate_figureS1",
            "fn":     "build_figure",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Figure S1: Correlation Decay Diagnostics",
            "save_output": str(o / "figS1.pdf"),
        },
        "sup_fig2": {
            "module": "figures.supplementary.generate_figureS2",
            "fn":     "build_figure",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Figure S2: Jacobian Spectral Density Evolution",
            "save_output": str(o / "figS2.pdf"),
        },
        "sup_fig3": {
            "module": "figures.supplementary.generate_figureS3",
            "fn":     "build_figure",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Figure S3: Finite-Size Scaling Collapse",
            "save_output": str(o / "figS3.pdf"),
        },
        "sup_fig4": {
            "module": "figures.supplementary.generate_figureS4",
            "fn":     "build_figure",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Figure S4: Stability Diagnostics",
            "save_output": str(o / "figS4.pdf"),
        },
        # ------------------------------------------------------------------
        # Supplementary Tables S1–S4
        # ------------------------------------------------------------------
        "sup_table1": {
            "module": "figures.supplementary.generate_tableS1",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Table S1: Depth-Scaling Statistics",
        },
        "sup_table2": {
            "module": "figures.supplementary.generate_tableS2",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Table S2: Hyperparameter Sweep Results",
        },
        "sup_table3": {
            "module": "figures.supplementary.generate_tableS3",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Table S3: Uncertainty Budget",
        },
        "sup_table4": {
            "module": "figures.supplementary.generate_tableS4",
            "fn":     "run",
            "kwargs": {"fast_track": ft},
            "group":  "supplementary",
            "description": "Supplementary Table S4: Metric Geometry Evolution",
        },
    }
    return registry


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_entry(key: str, spec: Dict) -> bool:
    """Import module, call the specified function, save if needed. Returns True on success."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'─'*60}")
    print(f"  {spec['description']}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(spec["module"])
        fn  = getattr(mod, spec["fn"])
        result = fn(**spec["kwargs"])

        # If fn returns a Figure (supplementary figs use build_figure pattern)
        if result is not None and hasattr(result, "savefig") and "save_output" in spec:
            Path(spec["save_output"]).parent.mkdir(parents=True, exist_ok=True)
            result.savefig(spec["save_output"], dpi=300, bbox_inches="tight")
            plt.close(result)

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
    group: Optional[str],
    results_root: str,
    output_dir: str,
    fast_track: bool,
    skip_on_error: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry    = _build_registry(Path(results_root), out, fast_track)
    to_generate = figures if figures else list(registry.keys())

    if group:
        to_generate = [k for k in to_generate
                       if registry.get(k, {}).get("group") == group]
        if not to_generate:
            print(f"No entries found for group '{group}'. "
                  f"Valid groups: manuscript, extended_data, supplementary")
            sys.exit(1)

    unknown = [k for k in to_generate if k not in registry]
    if unknown:
        print(f"Unknown keys: {unknown}")
        print(f"Valid keys: {list(registry.keys())}")
        sys.exit(1)

    mode_label = "FAST-TRACK [FAST_TRACK_UNVERIFIED]" if fast_track else "FULL QUALITY"
    print(f"\n{'='*60}")
    print(f"  RGP Figure Pipeline — {mode_label}")
    print(f"  Generating {len(to_generate)} item(s)")
    print(f"  Output: {out.resolve()}")
    print(f"{'='*60}")

    t_total         = time.perf_counter()
    passed, failed  = 0, 0

    for key in to_generate:
        success = _run_entry(key, registry[key])
        if success:
            passed += 1
        else:
            failed += 1
            if not skip_on_error:
                print("\nAborting (--no-skip-on-error).")
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
        description="Generate all RGP figures and tables (manuscript, extended data, supplementary)."
    )
    p.add_argument("--figures", nargs="+", default=None, metavar="KEY",
                   help="Specific keys to generate (e.g. fig1 fig3 ed_fig2 sup_fig1). "
                        "Default: all.")
    p.add_argument("--group", default=None,
                   choices=["manuscript", "extended_data", "supplementary"],
                   help="Generate only the specified group.")
    p.add_argument("--results-root", default="results",
                   help="Root directory for experiment results.")
    p.add_argument("--output", default="figures/out",
                   help="Output directory for generated figures/tables.")
    p.add_argument("--fast-track", action="store_true",
                   help="Use synthetic data (< 5 minutes, no GPU required).")
    p.add_argument("--no-skip-on-error", action="store_true",
                   help="Abort pipeline on first failure.")
    p.add_argument("--list", action="store_true",
                   help="List all available keys and exit.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        reg = _build_registry(Path("results"), Path("figures/out"), False)
        groups = {}
        for key, spec in reg.items():
            groups.setdefault(spec["group"], []).append((key, spec["description"]))
        for grp, items in groups.items():
            print(f"\n  {grp.upper()}")
            for key, desc in items:
                print(f"    {key:<14} {desc}")
        sys.exit(0)

    generate_all(
        figures=args.figures,
        group=args.group,
        results_root=args.results_root,
        output_dir=args.output,
        fast_track=args.fast_track,
        skip_on_error=not args.no_skip_on_error,
    )
