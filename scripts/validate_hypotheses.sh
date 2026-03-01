#!/usr/bin/env bash
# =============================================================================
# scripts/validate_hypotheses.sh
#
# Orchestrates H1, H2, H3 validation with statistical significance checks.
# Reports pass/fail per hypothesis with p-values and effect sizes.
#
# Usage: bash scripts/validate_hypotheses.sh [--fast-track]
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

FAST_TRACK=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast-track) FAST_TRACK="--fast-track"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== Hypothesis Validation Suite ==="
echo ""

FAILURES=0

run_validation() {
    local name="$1"
    local script="$2"
    local t0=$(date +%s)
    echo "--- ${name} ---"
    if python3 "$script" $FAST_TRACK --output "results/$(basename $(dirname $script))/"; then
        echo "[PASS] ${name} ($(( $(date +%s) - t0 ))s)"
    else
        echo "[FAIL] ${name}"
        FAILURES=$(( FAILURES + 1 ))
    fi
    echo ""
}

run_validation "H1: Scale Correspondence" \
    experiments/h1_scale_correspondence/run_h1_validation.py

run_validation "H2: Depth Scaling Law" \
    experiments/h2_depth_scaling/run_h2_validation.py

run_validation "H3: Multi-Scale Generalization" \
    experiments/h3_multiscale_generalization/run_h3_validation.py

echo "=== Results: $((3 - FAILURES))/3 hypotheses validated ==="
exit $FAILURES
