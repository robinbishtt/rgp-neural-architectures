set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"
if [ "$MODE" = "--fast-track" ]; then
    python3 ablation/run_all_ablations.py --fast-track --output results/ablation/
else
    python3 ablation/run_all_ablations.py --output results/ablation/
fi
