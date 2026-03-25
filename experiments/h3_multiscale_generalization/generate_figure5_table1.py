from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 and Table 1 (H3 Generalization) for the paper."
    )
    parser.add_argument("--results",      type=str,
                        default="results/h3/h3_results.json")
    parser.add_argument("--output-fig",   type=str,
                        default="figures/out/fig5.pdf")
    parser.add_argument("--output-table", type=str,
                        default="figures/out/table1.tex")
    parser.add_argument("--fast-track",   action="store_true")
    parser.add_argument("--dpi",          type=int, default=300)
    args = parser.parse_args()
    mod = importlib.import_module("figures.manuscript.generate_figure5")
    fig_args = [
        , args.output_fig,
        , args.output_table,
        , str(args.dpi),
    ]
    if args.fast_track:
        fig_args.append("--fast-track")
    else:
        fig_args += ["--results", args.results]
    sys.argv = ["generate_figure5.py"] + fig_args
    mod.main()
if __name__ == "__main__":
    main()