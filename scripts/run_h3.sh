#!/usr/bin/env bash
# Run H3 Multi-Scale Generalisation validation
# Full run: 6-8 hours on RTX 3090; fast-track: 3-5 minutes
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"
if [ "$MODE" = "--fast-track" ]; then
    python3 experiments/h3_multiscale_generalization/run_h3_validation.py --fast-track --results-dir results/h3/
else
    python3 experiments/h3_multiscale_generalization/run_h3_validation.py --results-dir results/h3/
fi
