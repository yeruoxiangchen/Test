#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${T2_REPAIR_GPU:-${CUDA_VISIBLE_DEVICES:-2}}"
exec /home/zjr/anaconda3/envs/trellis2/bin/python -B -m pose_conditioned_geometry_repair.dispatch "$@"
