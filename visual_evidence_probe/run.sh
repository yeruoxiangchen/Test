#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export CUDA_VISIBLE_DEVICES="${EVIDENCE_GPU:-0}"
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPCONV_ALGO=native
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR=/home/zjr/Tracker/visual_evidence_probe/outputs/torch_extensions
export MPLCONFIGDIR=/home/zjr/Tracker/visual_evidence_probe/outputs/matplotlib
exec /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m visual_evidence_probe.runner "$@"
