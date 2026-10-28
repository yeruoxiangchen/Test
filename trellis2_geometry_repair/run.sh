#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
task_store=/data/zjr/trellis2_geometry_repair
export TMPDIR="$task_store/tmp" XDG_CACHE_HOME="$task_store/cache"
export TORCH_HOME="$task_store/cache/torch" HF_HOME="$task_store/cache/huggingface"
export HF_HUB_CACHE="$task_store/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$task_store/cache/torch_extensions" MPLCONFIGDIR="$task_store/cache/matplotlib"
export CUDA_CACHE_PATH="$task_store/cache/cuda" TRITON_CACHE_DIR="$task_store/cache/triton"
export NUMBA_CACHE_DIR="$task_store/cache/numba" CUMM_CACHE_PATH="$task_store/cache/cumm"
export TORCHINDUCTOR_CACHE_DIR="$task_store/cache/torchinductor" IMAGEIO_USERDIR="$task_store/cache/imageio"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export SPARSE_CONV_BACKEND=flex_gemm ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/zjr/Tracker:/home/zjr/Tracker/TRELLIS2${PYTHONPATH:+:$PYTHONPATH}
task_python=/home/zjr/anaconda3/envs/trellis2/bin/python
action=${1:-help};if (( $# ));then shift;fi
export CUDA_VISIBLE_DEVICES=${T2_REPAIR_GPU:-6}
if [[ "${1:-}" == --gpu ]];then export CUDA_VISIBLE_DEVICES=${2:?GPU required};shift 2;fi
case "$action" in
  assets)
    mkdir -p "$TMPDIR"
    exec ionice -c 3 nice -n 10 "$task_python" -B -u -m trellis2_geometry_repair.assets "$@" ;;
  prepare|generate|smoke|fit|edit|evaluate|figures|pipeline|status|environment|report|compat|replay|converter_audit)
    exec "$task_python" -B -u -m "trellis2_geometry_repair.$action" "$@" ;;
  tests)
    mkdir -p "$task_store/outputs/native512_pilot_20261028/validation"
    task_log="$task_store/outputs/native512_pilot_20261028/validation/tests.log"
    if "$task_python" -B -m unittest trellis2_geometry_repair.test_core -v "$@" > "$task_log" 2>&1;then task_code=0;else task_code=$?;fi
    cat "$task_log";exit "$task_code" ;;
  launch)
    task_unit=${1:?unit required};shift
    if [[ ! "$task_unit" =~ ^trellis2-repair-[a-zA-Z0-9-]+$ ]];then exit 2;fi
    if [[ "${1:-}" == --gpu ]];then export CUDA_VISIBLE_DEVICES=${2:?GPU required};shift 2;fi
    task_logs="$task_store/outputs/native512_pilot_20261028/logs";mkdir -p "$task_logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --setenv="T2_REPAIR_GPU=$CUDA_VISIBLE_DEVICES" \
      --property=WorkingDirectory=/home/zjr/Tracker --property=Nice=10 --property=IOSchedulingClass=idle \
      --property="StandardOutput=append:$task_logs/$task_unit.log" \
      --property="StandardError=append:$task_logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/trellis2_geometry_repair/run.sh pipeline "$@" ;;
  *) printf '%s\n' 'Usage: bash trellis2_geometry_repair/run.sh {assets|prepare|tests|launch UNIT --gpu 6|status}';;
esac
