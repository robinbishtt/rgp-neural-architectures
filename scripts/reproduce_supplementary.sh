#!/usr/bin/env bash
# scripts/reproduce_supplementary.sh
#
# Reproduces all supplementary figures (S1-S4) and tables (S1-S4) in one shot.
# Requires: Python 3.10+, numpy, scipy, matplotlib (see requirements.txt).
#
# Fast-track mode (~2 min, synthetic data — no GPU required):
#   bash scripts/reproduce_supplementary.sh --fast-track
#
# Full quality mode (requires real experiment results in results/):
#   bash scripts/reproduce_supplementary.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

FAST_TRACK=0
OUTPUT_DIR="figures/out"
RESULTS_ROOT="results"

for arg in "$@"; do
    case "$arg" in
        --fast-track) FAST_TRACK=1 ;;
        --output=*)   OUTPUT_DIR="${arg#*=}" ;;
        --results=*)  RESULTS_ROOT="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--fast-track] [--output=DIR] [--results=DIR]"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "  RGP Supplementary Reproduction Script"
echo "  Mode:    $([ $FAST_TRACK -eq 1 ] && echo FAST-TRACK || echo FULL)"
echo "  Output:  $OUTPUT_DIR"
echo "============================================================"

FAST_FLAG=""
[ $FAST_TRACK -eq 1 ] && FAST_FLAG="--fast-track"

mkdir -p "$OUTPUT_DIR"
mkdir -p "results/supplementary"

echo ""
echo "--- Supplementary Figures ---"
python figures/generate_all.py \
    --group supplementary \
    --results-root "$RESULTS_ROOT" \
    --output "$OUTPUT_DIR" \
    $FAST_FLAG

echo ""
echo "============================================================"
echo "  Done. Outputs written to: $OUTPUT_DIR"
echo "  Supplementary tables in:  results/supplementary/"
echo "============================================================"
