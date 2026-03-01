#!/usr/bin/env bash
# scripts/reproduce_tables.sh — Generates all LaTeX tables from results/.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Generating tables ==="
python3 figures/manuscript/generate_figure5.py \
    --results results/h3/h3_results.json \
    --output figures/out/table1.tex
echo "Table 1 written to figures/out/table1.tex"
