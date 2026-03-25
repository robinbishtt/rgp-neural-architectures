set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---full}"
OUTPUT="${2:-figures/out/}"
if [ "$MODE" = "--fast-track" ]; then
    python3 figures/generate_all.py --fast-track --output "$OUTPUT"
else
    python3 figures/generate_all.py --results-root results/ --output "$OUTPUT"
fi
echo "Figures saved to: $OUTPUT"
