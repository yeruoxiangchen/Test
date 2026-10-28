#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/data/zjr/visual_evidence_probe/tmp XDG_CACHE_HOME=/data/zjr/visual_evidence_probe/cache
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export CUMM_CACHE_PATH="$XDG_CACHE_HOME/cumm" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export IMAGEIO_USERDIR="$XDG_CACHE_HOME/imageio" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 PYTHONPATH=/home/zjr/Tracker${PYTHONPATH:+:$PYTHONPATH}
task_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
action=${1:-status};if (( $# ));then shift;fi
case "$action" in
  launch)
    task_unit=${1:?unit};task_stage=${2:?stage};shift 2
    [[ "$task_unit" =~ ^visual-evidence-paired-[a-zA-Z0-9-]+$ ]] || exit 2
    task_logs=/data/zjr/visual_evidence_probe/outputs/paired_depth_diagnostic_20261028/logs
    mkdir -p "$task_logs"
    exec systemd-run --user --collect --service-type=exec --unit "$task_unit" \
      --property=WorkingDirectory=/home/zjr/Tracker \
      --property="StandardOutput=append:$task_logs/$task_unit.log" \
      --property="StandardError=append:$task_logs/$task_unit.log" \
      /usr/bin/bash /home/zjr/Tracker/visual_evidence_probe/run_paired_depth_diagnostic.sh "$task_stage" "$@" ;;
  tests)
    task_validation=/data/zjr/visual_evidence_probe/outputs/paired_depth_diagnostic_20261028/validation
    mkdir -p "$task_validation"
    if "$task_python" -B -m unittest visual_evidence_probe.experiments.paired_depth_diagnostic.test_pairing visual_evidence_probe.experiments.constrained_evidence.test_constraints -v > "$task_validation/tests.log" 2>&1;then task_status=0;else task_status=$?;fi
    cat "$task_validation/tests.log";exit "$task_status" ;;
  *) exec "$task_python" -B -u -m "visual_evidence_probe.experiments.paired_depth_diagnostic.$action" "$@" ;;
esac
