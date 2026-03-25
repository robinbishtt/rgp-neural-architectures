set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"
if [ "$MODE" = "--fast-track" ]; then
    python3 experiments/h2_depth_scaling/run_h2_validation.py --fast-track --results-dir results/h2/
else
    python3 experiments/h2_depth_scaling/run_h2_validation.py --results-dir results/h2/
fi
