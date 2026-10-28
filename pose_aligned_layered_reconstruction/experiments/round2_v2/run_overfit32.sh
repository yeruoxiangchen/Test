#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${V2_OVERFIT_GPU:-5}"
export CUDA_HOME=/usr/local/cuda-12.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=8.6 SPCONV_ALGO=native PYTHONFAULTHANDLER=1 V2_NPROC=1
v2_pkg=pose_aligned_layered_reconstruction
v2_run="$v2_pkg/experiments/round2_v2/run.sh"
v2_clean="$v2_pkg/outputs/round1_20260909/data_v2/train32.json"
v2_out="$v2_pkg/outputs/round2_v2_overfit32_20261017"
if [ -e "$v2_out" ]; then
  echo "Output already exists: $v2_out" >&2
  exit 1
fi
trap 'v2_status=$?; printf "[%s] exit_status=%s\n" "$(date -u +%FT%TZ)" "$v2_status"' EXIT
printf '[%s] stage=prepare_occlusion gpu=%s\n' "$(date -u +%FT%TZ)" "$CUDA_VISIBLE_DEVICES"
python -u -m pose_aligned_layered_reconstruction.data_v2.build_legacy_occlusion \
  --manifest "$v2_clean" --output_dir "$v2_out/occlusion" --max_objects 32 --variants 2
printf '[%s] stage=merge\n' "$(date -u +%FT%TZ)"
python -u -m pose_aligned_layered_reconstruction.data_v2.merge_manifests \
  --manifests "$v2_clean" "$v2_out/occlusion/manifest.json" --output "$v2_out/train96.json"
printf '[%s] stage=lari\n' "$(date -u +%FT%TZ)"
bash "$v2_run" lari --manifest "$v2_out/train96.json" --output_dir "$v2_out/lari" \
  --max_steps 2000 --grad_accum 2 --learning_rate 1e-4 --warmup_steps 50 --save_every 500 --p_uncond 0
printf '[%s] stage=ss\n' "$(date -u +%FT%TZ)"
bash "$v2_run" ss --manifest "$v2_out/train96.json" --init "$v2_out/lari/checkpoints/last.pt" \
  --output_dir "$v2_out/ss" --max_steps 2000 --grad_accum 2 --learning_rate 1e-4 \
  --geometry_lr_scale 0.2 --warmup_steps 50 --save_every 500 --p_uncond 0
printf '[%s] stage=complete\n' "$(date -u +%FT%TZ)"
