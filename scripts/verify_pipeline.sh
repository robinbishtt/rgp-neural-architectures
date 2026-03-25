set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/verify_pipeline.py
