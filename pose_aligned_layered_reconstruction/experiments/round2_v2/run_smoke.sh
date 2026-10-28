#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="${2:-${V2_SMOKE_GPU:-4}}"
export TORCH_CUDA_ARCH_LIST=8.6
export SPCONV_ALGO=native
v2_root=pose_aligned_layered_reconstruction/outputs/round2_v2_verified_20261017
v2_manifest=pose_aligned_layered_reconstruction/outputs/round1_20260909/data_v2/train2000.json
v2_python=/home/zjr/anaconda3/envs/reconviagen/bin/python
case "${1:-lari}" in
  lari)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage lari \
      --manifest "$v2_manifest" --output_dir "$v2_root/lari_current" --max_steps 2 \
      --save_every 1 --max_samples 2 --p_uncond 0
    ;;
  ss)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage ss \
      --manifest "$v2_manifest" --init "$v2_root/lari_current/checkpoints/last.pt" \
      --output_dir "$v2_root/ss" --max_steps 2 --save_every 1 --max_samples 2 --p_uncond 0
    ;;
  slat)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage slat \
      --manifest "$v2_manifest" --init "$v2_root/ss/checkpoints/last.pt" \
      --output_dir "$v2_root/slat_final" --max_steps 2 --save_every 1 --max_samples 2 --p_uncond 0 --trace_phases
    ;;
  render)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.render_scenes \
      --recipe pose_aligned_layered_reconstruction/experiments/round2_v2/configs/scene_smoke.json \
      --output_dir "$v2_root/full_scene_current"
    ;;
  encode)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.encode_ss_targets \
      --mesh "$v2_root/full_scene_current/target_canonical.obj" --output "$v2_root/full_scene_current/ss_target.npz"
    ;;
  cache)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.build_feature_cache \
      --scenes "$v2_root/full_scene_current/scene.json" --ss_targets "$v2_root/full_scene_current/ss_target.npz" \
      --output_dir "$v2_root/full_scene_cache"
    ;;
  legacy)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.build_legacy_occlusion \
      --manifest "$v2_manifest" --max_objects 1 --variants 1 --output_dir "$v2_root/legacy_occlusion"
    ;;
  infer)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.infer \
      --manifest "$v2_manifest" --checkpoint "$v2_root/ss/checkpoints/last.pt" \
      --slat_checkpoint "$v2_root/slat_final/checkpoints/last.pt" --output_dir "$v2_root/mesh" --max_samples 1
    ;;
  resume)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage lari \
      --manifest "$v2_manifest" --resume "$v2_root/lari_current/checkpoints/step_000001.pt" \
      --output_dir "$v2_root/lari_resume" --max_steps 2 --save_every 1 --max_samples 2 --p_uncond 0
    ;;
  dataset)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.data_v2.build_dataset \
      --assets pose_aligned_layered_reconstruction/experiments/round2_v2/configs/assets_smoke.json \
      --split train --variants 1 --max_objects 1 --output_dir "$v2_root/paired_dataset"
    ;;
  ddp)
    export CUDA_VISIBLE_DEVICES=6,7
    exec "$v2_python" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
      -m pose_aligned_layered_reconstruction.v2.train --stage lari --manifest "$v2_manifest" \
      --output_dir "$v2_root/lari_ddp_verified" --max_steps 1 --save_every 1 --max_samples 2 --p_uncond 0
    ;;
  new_data)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage ss \
      --manifest "$v2_root/mixed_new_data.json" --init "$v2_root/lari_current/checkpoints/last.pt" \
      --output_dir "$v2_root/ss_new_data" --max_steps 3 --save_every 3 --p_uncond 0
    ;;
  ss_isolated)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.train --stage ss \
      --manifest "$v2_root/mixed_new_data.json" --init "$v2_root/lari_current/checkpoints/last.pt" \
      --output_dir "$v2_root/ss_isolated" --max_steps 1 --save_every 1 --p_uncond 0
    ;;
  infer_isolated)
    exec "$v2_python" -u -m pose_aligned_layered_reconstruction.v2.infer \
      --manifest "$v2_manifest" --checkpoint "$v2_root/ss/checkpoints/last.pt" \
      --slat_checkpoint "$v2_root/slat_final/checkpoints/last.pt" --output_dir "$v2_root/mesh_isolated" --max_samples 1
    ;;
  *) echo 'usage: run_smoke.sh lari|ss|slat|render|encode|cache|legacy|infer|resume' >&2; exit 2 ;;
esac
