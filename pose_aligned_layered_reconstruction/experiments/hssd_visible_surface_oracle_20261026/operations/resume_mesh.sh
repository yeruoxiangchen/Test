#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
cd "$ROOT"
DATA_ROOT=/data/zjr/pose_aligned_layered_reconstruction
export TMPDIR="$DATA_ROOT/tmp" TORCH_HOME="$DATA_ROOT/cache/torch"
export HF_HOME="$DATA_ROOT/cache/huggingface" HF_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$DATA_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$DATA_ROOT/cache/matplotlib" XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"
export NUMBA_CACHE_DIR="$DATA_ROOT/cache/numba"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=flash_attn SPCONV_ALGO=native CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
exec /home/zjr/anaconda3/envs/reconviagen/bin/python "$HERE/resume_mesh.py" "$@"
