#!/usr/bin/env bash
# Dual H100: bash scripts/recap_training_eval.sh --gpus 0,1
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'uv is required; use the configured SuperMimic environment.' >&2; exit 1; }
exec env -u PYTHONPATH uv run --locked python -u scripts/recap_training_eval.py "$@"
