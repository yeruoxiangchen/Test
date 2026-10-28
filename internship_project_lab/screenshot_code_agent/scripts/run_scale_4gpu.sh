#!/usr/bin/env bash
set -euo pipefail
SCA_SCALE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_SCALE_ROOT"
export PYTHONPATH="$SCA_SCALE_ROOT/src"
exec "$SCA_SCALE_ROOT/.venv/bin/python" -u scripts/run_scale_4gpu.py "$@"
