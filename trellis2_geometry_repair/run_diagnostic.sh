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
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/zjr/Tracker:/home/zjr/Tracker/TRELLIS2${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${T2_DIAGNOSTIC_GPU:-6}
task_python=/home/zjr/anaconda3/envs/trellis2/bin/python
task_module=trellis2_geometry_repair.experiments.repair_diagnostic
task_out="$task_store/outputs/repair_diagnostic_20261028"
action=${1:-help};if (( $# ));then shift;fi
case "$action" in
  prepare|runner|evaluate|report|pipeline|status|replay|parallel|support_audit|finalize)
    exec "$task_python" -B -u -m "$task_module.$action" "$@" ;;
  tests)
    mkdir -p "$task_out/validation" "$TMPDIR"
    task_log="$task_out/validation/tests.log"
    if "$task_python" -B -m unittest trellis2_geometry_repair.test_core "$task_module.test_objective" -v > "$task_log" 2>&1;then task_code=0;else task_code=$?;fi
    cat "$task_log";exit "$task_code" ;;
  launch|launch_subset)
    task_entry=pipeline
    if [[ "$action" == launch_subset ]];then task_entry=parallel;fi
    task_unit=${1:?unit required};shift
    if [[ ! "$task_unit" =~ ^t2-diagnostic-[a-zA-Z0-9-]+$ ]];then exit 2;fi
    if [[ "${1:-}" == --gpu ]];then export CUDA_VISIBLE_DEVICES=${2:?GPU required};shift 2;fi
    mkdir -p "$task_out/logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --setenv="T2_DIAGNOSTIC_GPU=$CUDA_VISIBLE_DEVICES" \
      --property=WorkingDirectory=/home/zjr/Tracker --property=Nice=10 --property=IOSchedulingClass=idle \
      --property="StandardOutput=append:$task_out/logs/$task_unit.log" \
      --property="StandardError=append:$task_out/logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/trellis2_geometry_repair/run_diagnostic.sh "$task_entry" "$@" ;;
  *) printf '%s\n' 'Usage: bash trellis2_geometry_repair/run_diagnostic.sh {tests|prepare|runner|evaluate|report|launch UNIT --gpu 6|status}';;
esac
