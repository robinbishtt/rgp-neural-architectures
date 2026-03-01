"""
experiments/h2_depth_scaling/generate_figure4.py

Figure 4 generation for H2: Depth Scaling Law.

Delegates to figures/manuscript/generate_figure4.py, providing
experiment-local access for users working within the H2 pipeline.

Usage:
    python experiments/h2_depth_scaling/generate_figure4.py
    python experiments/h2_depth_scaling/generate_figure4.py --fast-track
    python experiments/h2_depth_scaling/generate_figure4.py \
        --results results/h2/h2_results.json \
        --output figures/out/fig4.pdf
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
        description="Generate Figure 4 (H2 Depth Scaling Law) for the paper."
    )
    parser.add_argument("--results",    type=str,
                        default="results/h2/h2_results.json")
    parser.add_argument("--output",     type=str,
                        default="figures/out/fig4.pdf")
    parser.add_argument("--fast-track", action="store_true")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    mod = importlib.import_module("figures.manuscript.generate_figure4")
    fig_args = ["--output", args.output, "--dpi", str(args.dpi)]
    if args.fast_track:
        fig_args.append("--fast-track")
    else:
        fig_args += ["--results", args.results]

    sys.argv = ["generate_figure4.py"] + fig_args
    mod.main()


if __name__ == "__main__":
    main()
