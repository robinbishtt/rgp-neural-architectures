#!/usr/bin/env bash
# reproduce.sh - One-command reproduction of all paper results.
#
# Usage:
#   bash reproduce.sh              Full reproduction (24-72 hours, requires GPU)
#   bash reproduce.sh --fast-track Fast-track (< 5 minutes, CPU-only)
#   bash reproduce.sh --h1         H1 only
#   bash reproduce.sh --h2         H2 only
#   bash reproduce.sh --h3         H3 only
#   bash reproduce.sh --figures    Figures only (requires existing results/)

set -euo pipefail
cd "$(dirname "$0")"

# ── Parse arguments ───────────────────────────────────────────────────
MODE="full"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast-track) MODE="fast" ;;
        --h1)         MODE="h1"   ;;
        --h2)         MODE="h2"   ;;
        --h3)         MODE="h3"   ;;
        --figures)    MODE="figures" ;;
        --ablations)  MODE="ablations" ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

START=$(date +%s)

echo ""
echo "========================================================"
echo "  Reproduction Pipeline"
echo "  Mode: $MODE"
echo "========================================================"
echo ""

# ── Step 0: Smoke test ────────────────────────────────────────────────
echo "[0/5] Verifying pipeline..."
python3 scripts/verify_pipeline.py
echo ""

# ── Step 0.5: Proof-of-life (fast-track only) ─────────────────────────
if [ "$MODE" = "fast" ]; then
    echo "[0.5/5] Proof-of-life: real small training run (depth=3, width=32)..."
    python3 scripts/proof_of_life_training.py --depth 3 --width 32 --epochs 3
    echo ""
fi

# ── Helper functions ──────────────────────────────────────────────────
run_h1() {
    echo "[1/5] H1: Scale Correspondence"
    if [ "$MODE" = "fast" ]; then
        python3 experiments/h1_scale_correspondence/run_h1_validation.py \
            --fast-track --results-dir results/h1/
    else
        python3 experiments/h1_scale_correspondence/run_h1_validation.py \
            --results-dir results/h1/
    fi
}

run_h2() {
    echo "[2/5] H2: Depth Scaling Law"
    if [ "$MODE" = "fast" ]; then
        python3 experiments/h2_depth_scaling/run_h2_validation.py \
            --fast-track --results-dir results/h2/
        python3 experiments/h2_depth_scaling/statistical_analysis.py --fast-track
    else
        python3 experiments/h2_depth_scaling/run_h2_validation.py \
            --results-dir results/h2/
        python3 experiments/h2_depth_scaling/statistical_analysis.py
    fi
}

run_h3() {
    echo "[3/5] H3: Multi-Scale Generalisation"
    if [ "$MODE" = "fast" ]; then
        python3 experiments/h3_multiscale_generalization/run_h3_validation.py \
            --fast-track --results-dir results/h3/
    else
        python3 experiments/h3_multiscale_generalization/run_h3_validation.py \
            --results-dir results/h3/
    fi
}

run_figures() {
    echo "[4/5] Generating figures"
    if [ "$MODE" = "fast" ]; then
        python3 figures/generate_all.py --fast-track --output figures/out/
    else
        python3 figures/generate_all.py --results-root results/ --output figures/out/
    fi
}

run_ablations() {
    echo "[5/5] Ablation studies"
    if [ "$MODE" = "fast" ]; then
        python3 ablation/run_all_ablations.py --fast-track --output results/ablation/
    else
        python3 ablation/run_all_ablations.py --output results/ablation/
    fi
}

# ── Execute ───────────────────────────────────────────────────────────
case "$MODE" in
    fast)
        run_h1; run_h2; run_h3; run_figures
        ;;
    h1)
        run_h1; run_figures
        ;;
    h2)
        run_h2; run_figures
        ;;
    h3)
        run_h3; run_figures
        ;;
    figures)
        run_figures
        ;;
    ablations)
        run_ablations
        ;;
    full)
        run_h1; run_h2; run_h3; run_figures; run_ablations
        ;;
esac

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "========================================================"
echo "  Reproduction complete in ${ELAPSED}s"
echo ""
if [ "$MODE" = "fast" ]; then
    echo "  NOTE: fast-track outputs are tagged [FAST_TRACK_UNVERIFIED]."
    echo "  For quantitative verification of paper claims, run:"
    echo "    bash reproduce.sh"
fi
echo "  Results:  results/"
echo "  Figures:  figures/out/"
echo "========================================================"
