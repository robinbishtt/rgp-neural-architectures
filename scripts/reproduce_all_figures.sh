#!/usr/bin/env bash
# scripts/reproduce_all_figures.sh - Regenerates all figures from results/.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Generating all figures ==="
python3 figures/generate_all.py --results-root results/ --output figures/out/
echo "Figures written to figures/out/"
