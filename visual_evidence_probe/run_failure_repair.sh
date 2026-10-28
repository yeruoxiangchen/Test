#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/data/zjr/visual_evidence_probe/tmp
export XDG_CACHE_HOME=/data/zjr/visual_evidence_probe/cache
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" CUMM_CACHE_PATH="$XDG_CACHE_HOME/cumm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" IMAGEIO_USERDIR="$XDG_CACHE_HOME/imageio"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-12.6 CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=/home/zjr/Tracker:/data/zjr/visual_evidence_probe/vendor/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}
task_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
task_da3_python=/data/zjr/visual_evidence_probe/envs/da3/bin/python
task_root=/data/zjr/visual_evidence_probe/outputs/failure_repair_20261028
action=${1:-help}
if (( $# )); then shift; fi
case "$action" in
  generate|regions)
    exec "$task_python" -B -u -m visual_evidence_probe.experiments.failure_repair.engine "$action" "$@" ;;
  depth)
    exec "$task_da3_python" -B -u -m visual_evidence_probe.experiments.failure_repair.engine depth "$@" ;;
  prepare|run|evaluate|report|figures|check|pipeline|status|closeout)
    exec "$task_python" -B -u -m "visual_evidence_probe.experiments.failure_repair.$action" "$@" ;;
  tests)
    mkdir -p "$task_root/validation"
    if "$task_python" -B -m unittest visual_evidence_probe.experiments.failure_repair.test_protocol -v > "$task_root/validation/tests.log" 2>&1; then task_code=0; else task_code=$?; fi
    cat "$task_root/validation/tests.log"
    exit "$task_code" ;;
  launch)
    task_unit=${1:?unit required}; shift
    if [[ ! "$task_unit" =~ ^visual-evidence-failure-[a-zA-Z0-9-]+$ ]]; then exit 2; fi
    mkdir -p "$task_root/logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --property=WorkingDirectory=/home/zjr/Tracker --property=Nice=10 --property=IOSchedulingClass=idle \
      --property="StandardOutput=append:$task_root/logs/$task_unit.log" \
      --property="StandardError=append:$task_root/logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/visual_evidence_probe/run_failure_repair.sh pipeline "$@" ;;
  *) printf '%s\n' 'Usage: bash visual_evidence_probe/run_failure_repair.sh {prepare|tests|launch UNIT --gpu 0|status|report}';;
esac
