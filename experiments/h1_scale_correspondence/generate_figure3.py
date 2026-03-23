"""
experiments/h1_scale_correspondence/generate_figure3.py

Figure 3 generation for H1: Scale Correspondence Validation.

This script loads H1 validation results and delegates to the canonical
figure generator in figures/manuscript/generate_figure3.py. It exists
here to provide experiment-local access to figure generation without
requiring users to navigate to the figures/ directory.

Usage:
    python experiments/h1_scale_correspondence/generate_figure3.py
    python experiments/h1_scale_correspondence/generate_figure3.py --fast-track
    python experiments/h1_scale_correspondence/generate_figure3.py \
        --results results/h1/h1_results.json \
        --output figures/out/fig3.pdf
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 (H1 Scale Correspondence) for the paper."
    )
    parser.add_argument("--results",    type=str,
                        default="results/h1/h1_results.json",
                        help="Path to H1 results JSON file.")
    parser.add_argument("--output",     type=str,
                        default="figures/out/fig3.pdf",
                        help="Output PDF path.")
    parser.add_argument("--fast-track", action="store_true",
                        help="Use synthetic data for pipeline verification.")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    # Delegate to canonical generator
    mod = importlib.import_module("figures.manuscript.generate_figure3")
    fig_args = [
        "--output",   args.output,
        "--dpi",      str(args.dpi),
    ]
    if args.fast_track:
        fig_args.append("--fast-track")
    else:
        fig_args += ["--results", args.results]

    sys.argv = ["generate_figure3.py"] + fig_args
    mod.main()


if __name__ == "__main__":
    main()
 