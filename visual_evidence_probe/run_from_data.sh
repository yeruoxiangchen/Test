#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data/zjr/visual_evidence_probe/tmp
export TORCH_HOME=/data/zjr/visual_evidence_probe/cache/torch
export HF_HOME=/data/zjr/visual_evidence_probe/cache/huggingface
export HF_HUB_CACHE=/data/zjr/visual_evidence_probe/cache/huggingface/hub
export TORCH_EXTENSIONS_DIR=/data/zjr/visual_evidence_probe/cache/torch_extensions
export MPLCONFIGDIR=/data/zjr/visual_evidence_probe/cache/matplotlib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /home/zjr/anaconda3/envs/reconviagen/bin/python -u -m visual_evidence_probe.analysis_scripts.data_entrypoint "$@"
