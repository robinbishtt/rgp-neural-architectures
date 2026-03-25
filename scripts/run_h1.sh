set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"
if [ "$MODE" = "--fast-track" ]; then
    python3 experiments/h1_scale_correspondence/run_h1_validation.py --fast-track --results-dir results/h1/
else
    python3 experiments/h1_scale_correspondence/run_h1_validation.py --results-dir results/h1/
fi
