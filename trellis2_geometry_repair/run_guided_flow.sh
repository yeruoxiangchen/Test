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
export CUDA_VISIBLE_DEVICES=${T2_GUIDED_GPU:-4} T2_LOCAL_PROFILE=v2
export T2_GUIDED_PROFILE=${T2_GUIDED_PROFILE:-fixed}
task_python=/home/zjr/anaconda3/envs/trellis2/bin/python
task_module=trellis2_geometry_repair.experiments.guided_flow
task_out="$task_store/outputs/guided_flow_v3_fixed_20261029"
if [[ "$T2_GUIDED_PROFILE" == initial ]];then task_out="$task_store/outputs/guided_flow_v3_20261029";fi
action=${1:-help};if (( $# ));then shift;fi
if [[ "$T2_GUIDED_PROFILE" == initial && "$action" != status ]];then printf '%s\n' 'Initial preflight is frozen; use its source archive to reproduce it.';exit 2;fi
case "$action" in
  prepare|smoke|generation|selection|evaluate|report|pipeline|status|paired_audit)
    exec "$task_python" -B -u -m "$task_module.$action" "$@" ;;
  tests)
    mkdir -p "$task_out/validation" "$TMPDIR"
    if "$task_python" -B -m unittest "$task_module.test_guidance" -v > "$task_out/validation/tests.log" 2>&1;then task_code=0;else task_code=$?;fi
    cat "$task_out/validation/tests.log";exit "$task_code" ;;
  launch)
    task_unit=${1:?unit required};shift
    if [[ ! "$task_unit" =~ ^t2-guided-[a-zA-Z0-9-]+$ ]];then exit 2;fi
    if [[ "${1:-}" == --gpu ]];then export CUDA_VISIBLE_DEVICES=${2:?GPU required};shift 2;fi
    mkdir -p "$task_out/logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --setenv="T2_GUIDED_GPU=$CUDA_VISIBLE_DEVICES" \
      --setenv="T2_GUIDED_PROFILE=$T2_GUIDED_PROFILE" \
      --property=WorkingDirectory=/home/zjr/Tracker --property=Nice=10 --property=IOSchedulingClass=idle \
      --property="StandardOutput=append:$task_out/logs/$task_unit.log" \
      --property="StandardError=append:$task_out/logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/trellis2_geometry_repair/run_guided_flow.sh pipeline "$@" ;;
  *) printf '%s\n' 'Usage: bash trellis2_geometry_repair/run_guided_flow.sh {tests|prepare|smoke|launch UNIT --gpu 4|status}';;
esac
