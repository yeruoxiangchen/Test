#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONFAULTHANDLER=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
v2_stage="${1:?usage: run.sh lari|ss|slat|infer|dataset [arguments]}"
shift
v2_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
case "$v2_stage" in
  lari|ss|slat)
    if [ "${V2_NPROC:-1}" -gt 1 ]; then
      exec "$v2_python" -u -m torch.distributed.run --standalone --nproc_per_node="$V2_NPROC" \
        -m pose_aligned_layered_reconstruction.v2.train --stage "$v2_stage" "$@"
    fi
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage "$v2_stage" "$@"
    ;;
  infer)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.infer "$@"
    ;;
  dataset)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.build_dataset "$@"
    ;;
  *) echo 'unknown stage' >&2; exit 2 ;;
esac
