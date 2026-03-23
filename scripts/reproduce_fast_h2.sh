#!/usr/bin/env bash
# scripts/reproduce_fast_h2.sh - Fast-track H2 (Depth Scaling) only.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/reproduce_fast.sh --hypothesis h2
