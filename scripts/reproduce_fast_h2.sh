set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/reproduce_fast.sh --hypothesis h2
