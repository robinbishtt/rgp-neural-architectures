#!/usr/bin/env bash
# =============================================================================
# scripts/reproduce_fast.sh
#
# Fast-track verification pipeline: L=10, width=64, 2 epochs.
# Completes in 3-5 minutes on any hardware (CPU included).
# All outputs tagged [FAST_TRACK_UNVERIFIED].
#
# Usage:
#   bash scripts/reproduce_fast.sh
#   bash scripts/reproduce_fast.sh --hypothesis h1
#   bash scripts/reproduce_fast.sh --hypothesis h2
#   bash scripts/reproduce_fast.sh --hypothesis h3
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

HYPOTHESIS="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --hypothesis) HYPOTHESIS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

START_TIME=$(date +%s)
echo "=== Fast-Track Verification (hypothesis=${HYPOTHESIS}) ==="
echo ""

run_h1() {
    echo "[H1] Scale Correspondence — fast-track..."
    python3 experiments/h1_scale_correspondence/run_h1_validation.py \
        --fast-track --output results/h1/
    echo "[H1] Done"
}

run_h2() {
    echo "[H2] Depth Scaling — fast-track..."
    python3 experiments/h2_depth_scaling/run_h2_validation.py \
        --fast-track --output results/h2/
    echo "[H2] Done"
}

run_h3() {
    echo "[H3] Multi-Scale Generalization — fast-track..."
    python3 experiments/h3_multiscale_generalization/run_h3_validation.py \
        --fast-track --output results/h3/
    echo "[H3] Done"
}

run_figures() {
    echo "[Figures] Generating figures from fast-track results..."
    python3 figures/generate_all.py --fast-track --output figures/out/
    echo "[Figures] Done"
}

case "$HYPOTHESIS" in
    all) run_h1; run_h2; run_h3; run_figures ;;
    h1)  run_h1; run_figures ;;
    h2)  run_h2; run_figures ;;
    h3)  run_h3; run_figures ;;
    *)   echo "Unknown hypothesis: $HYPOTHESIS (must be h1|h2|h3|all)" >&2; exit 1 ;;
esac

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=== Fast-track complete in ${ELAPSED}s ==="
echo "Outputs: results/  figures/out/"
echo "NOTE: Outputs are [FAST_TRACK_UNVERIFIED]. For full verification, run: make reproduce_all"
