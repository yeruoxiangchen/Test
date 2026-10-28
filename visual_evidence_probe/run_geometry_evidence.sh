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
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CUMM_CACHE_PATH="$XDG_CACHE_HOME/cumm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-12.6
task_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
action=${1:-help}
if (( $# )); then shift; fi
case "$action" in
  launch)
    task_unit=${1:?unit name required}; shift
    if [[ ! "$task_unit" =~ ^visual-evidence-geometry-[a-zA-Z0-9-]+$ ]]; then exit 2; fi
    task_logs=/data/zjr/visual_evidence_probe/outputs/geometry_evidence_20261027/logs
    mkdir -p "$task_logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --property=WorkingDirectory=/home/zjr/Tracker \
      --property="StandardOutput=append:$task_logs/$task_unit.log" \
      --property="StandardError=append:$task_logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/visual_evidence_probe/run_geometry_evidence.sh run "$@" ;;
  run) exec "$task_python" -B -u -m visual_evidence_probe.experiments.geometry_evidence.runner "$@" ;;
  report|figures|closeout) exec "$task_python" -B -u -m "visual_evidence_probe.experiments.geometry_evidence.$action" "$@" ;;
  tests)
    task_validation=/data/zjr/visual_evidence_probe/outputs/geometry_evidence_20261027/validation
    mkdir -p "$task_validation"
    if "$task_python" -B -m unittest visual_evidence_probe.experiments.geometry_evidence.test_protocol -v "$@" > "$task_validation/tests.log" 2>&1; then
      task_status=0
    else
      task_status=$?
    fi
    cat "$task_validation/tests.log"
    exit "$task_status" ;;
  *) printf '%s\n' 'Usage: bash visual_evidence_probe/run_geometry_evidence.sh {launch UNIT|run|report|figures|closeout|tests} [arguments]' ;;
esac
