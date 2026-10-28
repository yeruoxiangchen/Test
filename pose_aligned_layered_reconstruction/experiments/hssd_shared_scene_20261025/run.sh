#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${HSSD_SHARED_GPU:-3}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=flash_attn SPCONV_ALGO=native
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
case "${1:-status}" in
  gena-download|start-gena-download) unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE ;;
  *) export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ;;
esac
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python
MODULE=pose_aligned_layered_reconstruction.experiments.hssd_shared_scene_20261025
case "${1:-status}" in
  prepare) exec "$PYTHON" -m "$MODULE.prepare" ;;
  external-status) exec "$PYTHON" -m "$MODULE.external" audit ;;
  external-check) exec "$PYTHON" -m "$MODULE.external" importcheck ;;
  audit-data) exec "$PYTHON" -m "$MODULE.cache" --audit-only ;;
  test) exec "$PYTHON" -m unittest "$MODULE.test_contracts" -v ;;
  start) exec "$PYTHON" -m "$MODULE.runner" start-all ;;
  *) exec "$PYTHON" -m "$MODULE.runner" "${1:-status}" ;;
esac
