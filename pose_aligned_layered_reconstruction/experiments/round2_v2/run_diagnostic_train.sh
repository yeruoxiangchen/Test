#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
v2_arm="${1:?arm required}"
v2_gpu="${2:?GPU index required}"
case "$v2_gpu" in [0-7]) ;; *) echo 'GPU must be 0..7' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="$v2_gpu"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 V2_NPROC=1
v2_source=pose_aligned_layered_reconstruction/outputs/round2_v2_overfit32_20261017
v2_output=pose_aligned_layered_reconstruction/outputs/round2_v2_diagnostics_20261018
v2_extra=()
v2_steps=500
case "$v2_arm" in
  smoke) v2_steps=2 ;;
  raw_joint) ;;
  raw_frozen) v2_extra+=(--freeze_geometry --cache_frozen_geometry) ;;
  raw_frozen_v1recipe) v2_extra+=(--freeze_geometry --cache_frozen_geometry --t_sampling logit_normal --lr_schedule constant) ;;
  raw_frozen_v1recipe_tol004) v2_extra+=(--freeze_geometry --cache_frozen_geometry --t_sampling logit_normal --lr_schedule constant --ss_absolute_tolerance .04) ;;
  *) echo 'unknown diagnostic arm' >&2; exit 2 ;;
esac
if [ -e "$v2_output/$v2_arm" ]; then
  echo "Diagnostic arm already exists: $v2_output/$v2_arm" >&2
  exit 1
fi
mkdir -p "$v2_output/logs"
exec bash pose_aligned_layered_reconstruction/experiments/round2_v2/run.sh ss \
  --manifest "$v2_source/train96.json" --init "$v2_source/lari/checkpoints/step_002000.pt" \
  --init_weights raw --output_dir "$v2_output/$v2_arm" --max_steps "$v2_steps" \
  --grad_accum 2 --learning_rate 1e-4 --geometry_lr_scale .2 --warmup_steps 50 \
  --lr_schedule_steps 2000 --save_every 250 --p_uncond 0 --seed 20261017 \
  "${v2_extra[@]}" > "$v2_output/logs/$v2_arm.log" 2>&1
