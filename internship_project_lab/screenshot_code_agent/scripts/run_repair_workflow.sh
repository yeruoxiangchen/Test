#!/usr/bin/env bash
set -euo pipefail
SCA_WORKFLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_WORKFLOW_ROOT"
export PYTHONPATH="$SCA_WORKFLOW_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$SCA_WORKFLOW_ROOT/.venv/bin/python" -u scripts/run_repair_workflow.py "$@"
