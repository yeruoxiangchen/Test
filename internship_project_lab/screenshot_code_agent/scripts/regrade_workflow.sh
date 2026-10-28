#!/usr/bin/env bash
set -euo pipefail
SCA_REGRADE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCA_REGRADE_CACHE=/data/zjr/screenshot_code_agent/grading_runtime_20261028
export TMPDIR="$SCA_REGRADE_CACHE/tmp"
export TORCH_HOME="$SCA_REGRADE_CACHE/torch"
export HF_HOME="$SCA_REGRADE_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$SCA_REGRADE_CACHE/torch_extensions"
export MPLCONFIGDIR="$SCA_REGRADE_CACHE/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCA_REGRADE_ROOT/src"
export CUDA_VISIBLE_DEVICES=""
if [[ "${1:-}" != status ]]; then
    mkdir -p "$TMPDIR" "$TORCH_HOME" "$HF_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"
fi
cd "$SCA_REGRADE_ROOT"
exec /usr/bin/nice -n 10 /usr/bin/ionice -c 2 -n 7 \
    "$SCA_REGRADE_ROOT/.venv/bin/python" -B -u scripts/regrade_workflow.py "$@"
