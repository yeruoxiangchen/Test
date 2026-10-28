#!/usr/bin/env bash
set -euo pipefail
SCA_COMPARE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_COMPARE_ROOT"
export PYTHONPATH="$SCA_COMPARE_ROOT/src"
exec "$SCA_COMPARE_ROOT/.venv/bin/python" -u scripts/run_v6_compare.py "$@"
