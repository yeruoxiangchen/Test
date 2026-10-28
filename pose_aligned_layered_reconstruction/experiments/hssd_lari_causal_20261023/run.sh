#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export CUDA_VISIBLE_DEVICES="${HSSD_CAUSAL_GPU:-0}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
  echo 'HSSD_CAUSAL_GPU must be one physical GPU index' >&2
  exit 2
fi
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPCONV_ALGO=native
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m \
  pose_aligned_layered_reconstruction.experiments.hssd_lari_causal_20261023.runner "$@"
