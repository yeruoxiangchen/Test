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
export IMAGEIO_USERDIR="$XDG_CACHE_HOME/imageio"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-12.6
export PYTHONPATH=/home/zjr/Tracker:/data/zjr/visual_evidence_probe/vendor/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}
task_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
task_da3_python=/data/zjr/visual_evidence_probe/envs/da3/bin/python
action=${1:-help}
if (( $# )); then shift; fi
case "$action" in
  setup)
    "$task_da3_python" -m pip install --disable-pip-version-check 'numpy==1.26.4' 'omegaconf==2.3.0' evo e3nn 'moviepy==1.0.3' pillow-heif pycolmap addict 'opencv-python-headless==4.11.0.86'
    exec "$task_da3_python" -B -u -m visual_evidence_probe.experiments.dense_evidence.da3_setup "$@" ;;
  download)
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
    exec "$task_da3_python" -B -u -m visual_evidence_probe.experiments.dense_evidence.da3_setup "$@" ;;
  depth)
    exec "$task_da3_python" -B -u -m "visual_evidence_probe.experiments.dense_evidence.$action" "$@" ;;
  prepare|generate|regions|depth_report|evaluate|run|report|figures|closeout|pipeline|status)
    exec "$task_python" -B -u -m "visual_evidence_probe.experiments.dense_evidence.$action" "$@" ;;
  launch)
    task_unit=${1:?unit required}; task_stage=${2:?stage required}; shift 2
    if [[ ! "$task_unit" =~ ^visual-evidence-dense-[a-zA-Z0-9-]+$ ]]; then exit 2; fi
    task_logs=/data/zjr/visual_evidence_probe/outputs/dense_evidence_20261027/logs
    mkdir -p "$task_logs"
    task_proxy_args=()
    for task_env_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
      if [[ -v "$task_env_name" ]]; then task_proxy_args+=("--setenv=$task_env_name=${!task_env_name}"); fi
    done
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      "${task_proxy_args[@]}" \
      --property=WorkingDirectory=/home/zjr/Tracker \
      --property="StandardOutput=append:$task_logs/$task_unit.log" \
      --property="StandardError=append:$task_logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/visual_evidence_probe/run_dense_evidence.sh "$task_stage" "$@" ;;
  tests)
    task_validation=/data/zjr/visual_evidence_probe/outputs/dense_evidence_20261027/validation
    mkdir -p "$task_validation"
    if "$task_python" -B -m unittest visual_evidence_probe.experiments.dense_evidence.test_protocol -v "$@" > "$task_validation/tests.log" 2>&1; then task_status=0; else task_status=$?; fi
    cat "$task_validation/tests.log"
    exit "$task_status" ;;
  *) printf '%s\n' 'Usage: bash visual_evidence_probe/run_dense_evidence.sh {setup|download|prepare|generate|depth|depth_report|launch UNIT STAGE|run|report|figures|closeout|tests} [arguments]' ;;
esac
