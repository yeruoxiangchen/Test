#!/usr/bin/env bash
set -euo pipefail
SCA_HARNESS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_HARNESS_ROOT"
export PYTHONPATH="$SCA_HARNESS_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$SCA_HARNESS_ROOT/.venv/bin/python" -u scripts/run_harness_compare.py "$@"
