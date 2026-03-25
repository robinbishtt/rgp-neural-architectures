set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Generating all figures ==="
python3 figures/generate_all.py --results-root results/ --output figures/out/
echo "Figures written to figures/out/"
