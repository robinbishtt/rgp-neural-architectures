#!/usr/bin/env bash
# scripts/reproduce_fast_h3.sh - Fast-track H3 (Multi-Scale Generalization) only.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/reproduce_fast.sh --hypothesis h3
