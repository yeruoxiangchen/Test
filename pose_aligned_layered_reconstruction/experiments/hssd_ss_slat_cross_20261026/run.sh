#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${HSSD_CROSS_GPU:-4}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=flash_attn SPCONV_ALGO=native CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python
ACTION="${1:-status}"
if (($#)); then
  shift
fi
exec "$PYTHON" "$HERE/runner.py" "$ACTION" "$@"
