#!/usr/bin/env bash
set -euo pipefail
SCA_EVAL_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SCA_LIMIT="${SCA_LIMIT:-0}"
exec bash "$SCA_EVAL_PROJECT_DIR/scripts/run_local_smoke.sh"
