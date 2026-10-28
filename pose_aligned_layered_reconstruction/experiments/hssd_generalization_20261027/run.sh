#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
cd "$ROOT"
DATA_ROOT=/data/zjr/pose_aligned_layered_reconstruction
export TMPDIR="$DATA_ROOT/tmp" TORCH_HOME="$DATA_ROOT/cache/torch"
export HF_HOME="$DATA_ROOT/cache/huggingface" HF_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$DATA_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$DATA_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$DATA_ROOT/cache/numba"
export XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6
export ATTN_BACKEND=flash_attn SPCONV_ALGO=native CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/zjr/anaconda3/envs/reconviagen/bin/python -B "$HERE/runner.py" "${1:-status}"
