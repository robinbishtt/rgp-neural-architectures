set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Generating Extended Data figures ==="
python3 figures/generate_all.py \
    --figures ed_fig1 ed_fig2 ed_fig3 ed_fig4 ed_fig5 ed_fig6 \
    --results-root results/ --output figures/out/
echo "Extended Data figures written to figures/out/"
