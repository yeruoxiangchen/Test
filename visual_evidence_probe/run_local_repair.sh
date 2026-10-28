#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/data/zjr/visual_evidence_probe/tmp
export XDG_CACHE_HOME=/data/zjr/visual_evidence_probe/cache
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export CUMM_CACHE_PATH="$XDG_CACHE_HOME/cumm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
task_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
action=${1:-help}
if (( $# )); then shift; fi
case "$action" in
  test) exec "$task_python" -B -m unittest visual_evidence_probe.experiments.local_repair.test_repair -v "$@" ;;
  strict) exec "$task_python" -B -u -m visual_evidence_probe.experiments.local_repair.runner run "$@" ;;
  cropped) exec "$task_python" -B -u -m visual_evidence_probe.analysis_scripts.local_repair_cropped run "$@" ;;
  latent) exec "$task_python" -B -u -m visual_evidence_probe.analysis_scripts.local_repair_latent_batch "$@" ;;
  controls|figures|finalize) exec "$task_python" -B -u -m visual_evidence_probe.analysis_scripts.local_repair_review "$action" "$@" ;;
  latent-report) exec "$task_python" -B -u -m visual_evidence_probe.analysis_scripts.local_repair_latent_report "$@" ;;
  *) printf '%s\n' 'Usage: bash visual_evidence_probe/run_local_repair.sh {test|strict|cropped|latent|controls|figures|finalize|latent-report} [arguments]' ;;
esac
