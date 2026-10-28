# Pose-Constrained TRELLIS.2

This is the clean, geometry-only migration of the pose-constrained multiview
method to the official TRELLIS.2 backend.  It is a new training line: the
official TRELLIS.2 SS and Shape-512 Flow checkpoints are the frozen bases;
the old 30K SS/SLat checkpoints are neither loaded nor distilled.

The implementation is independent of the maintained paper package and the
archived experimental package.  Its only source-level runtime dependency in
this repository is the sibling `../TRELLIS2` checkout (or the path specified
by `TRELLIS2_ROOT`).

## Environment

Use an environment built with the official TRELLIS.2 `--basic --flash-attn
--cumesh --o-voxel --flexgemm` components, then install
[`requirements-project.txt`](requirements-project.txt).  `peft` supplies the
fresh attention LoRA modules and `objaverse` is used only by the native-target
download adapter.  The local `trellis2` environment already contains the
compiled TRELLIS.2 CUDA extensions, but in this session it still lacks those
two lightweight additions.  The `reconviagen` environment is sufficient for
CPU contract tests and Flow construction, but lacks `o_voxel` and therefore is
not an end-to-end inference environment.

## Frozen scope

| Part | This migration |
| --- | --- |
| Image encoder | DINOv3 ViT-L/16, 512 px, 5 prefix + 32x32 patch tokens |
| Sparse structure | TRELLIS.2 SS Flow, 8x16x16x16 |
| Shape | TRELLIS.2 Shape-SLat 512 Flow, active 32³ coordinates, 32 channels |
| Shape token bound | official Shape-512 filter: at most 8192 active tokens/object |
| Decoder | TRELLIS.2 Flexible Dual Grid geometry decoder at 512 |
| Pose condition | all selected views projected onto native SS/Shape queries |
| View fusion | learned mean/variance aggregation from the current method |
| Injection | an independent zero-initialized projection before every one of 30 blocks |
| Trainable backbone surface | attention LoRA, rank 8 / alpha 16 by default |
| Initialization | official TRELLIS.2 base + fresh zero adapters/LoRA |
| Texture | excluded; no texture Flow or texture decoder is loaded |

The native 2D cross-attention context defaults to the complete 1029-token
sequence of the first selected view.  This preserves the official single-view
token distribution while all selected views still enter the explicit
pose-projected 3D condition.  `--context-mode concat` is retained as an
explicit multiview-context ablation.

## Native geometry targets

Use the official TRELLIS.2 data toolkit from the original raw assets; do not
convert or relabel an old TRELLIS latent cache.  The current local frozen 30K
root has posed renders and incompatible TRELLIS-v1 `lh-slats`, but not raw
assets or native T2 targets.  [`DATA_PREPARATION.md`](DATA_PREPARATION.md)
records the exact frozen UID export, raw-asset recovery, Flexible Dual Grid
512/1024 conversion, native Shape/SS encoding, posed-cache materialization,
and strict finalization commands.  It also supplies the missing released
`datasets.ObjaverseXL` toolkit interface locally without modifying TRELLIS2.

## Posed-image cache

Create a raw-view specification following
[`examples/raw_views.example.json`](examples/raw_views.example.json).  Each
sample contains ordered image paths, optional masks, pixel-space intrinsics,
and either `w2c` or `c2w` extrinsics.  `camera_forward_sign` binds the camera
forward axis to positive projection depth; `camera_image_y_sign` is `+1` for
OpenCV-style camera `+Y` down and `-1` for OpenGL/Blender-style camera `+Y`
up.  Masks may also come from image alpha.

```bash
cd /home/zjr/Tracker

python -m pose_constrained_trellis2.prepare_pose_cache \
  --spec /path/to/raw_views.json \
  --output-dir /data/pose_t2_cache \
  --ss-target-dir "$DATA_ROOT/ss_latents/ss_enc_conv3d_16l8_fp16_64" \
  --shape-target-dir "$DATA_ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
```

The cache builder applies one foreground-aware square crop to both RGB and K,
encodes the complete DINOv3 sequence, hashes its files, and rejects incompatible
native target dimensions before training.  It never imports old project code.

## Fresh 30K training

The two stages are intentionally trained separately.  The commands below use
the frozen defaults: global batch eight with eight processes, random 1--8 view
subsets, BF16, gradient checkpointing, EMA 0.9995, and 30,000 optimizer steps.

```bash
cd /home/zjr/Tracker

torchrun --standalone --nproc_per_node=8 \
  -m pose_constrained_trellis2.train \
  --stage ss \
  --manifest /data/pose_t2_cache/manifest.json \
  --output-dir /data/pose_t2_ss_30k

torchrun --standalone --nproc_per_node=8 \
  -m pose_constrained_trellis2.train \
  --stage shape \
  --manifest /data/pose_t2_cache/manifest.json \
  --output-dir /data/pose_t2_shape_30k
```

Every fresh run performs a stock-preservation audit before optimization.  At
step zero, the public adapted output must equal the frozen official Flow
exactly while gradients through every zero-initialized block projection remain
live.  Normal training avoids a second frozen-stock forward after this anchor;
`--stock-eval-every N` enables that diagnostic only at an explicit interval.
Checkpoints contain only trainable adapters/LoRA (raw and EMA), optimizer state,
per-rank RNG state, and frozen model/data identities.  `last.pt` is an atomic
hard link to the latest numbered checkpoint rather than a duplicate copy.

## Two-stage geometry inference

```bash
python -m pose_constrained_trellis2.infer \
  --manifest /data/pose_t2_cache/manifest.json \
  --sample-index 0 \
  --views 0,1,2,3 \
  --ss-checkpoint /data/pose_t2_ss_30k/checkpoints/step_030000.pt \
  --shape-checkpoint /data/pose_t2_shape_30k/checkpoints/step_030000.pt \
  --weights ema \
  --output /data/pose_t2_result/object.ply
```

Inference loads and releases SS Flow, SS decoder, Shape Flow, and Shape decoder
in sequence, so both 1.3B Flows are not resident together.  The result is an
untextured PLY or OBJ plus a JSON report binding views, seed, checkpoints,
samplers, support size, and mesh size.

## Verification

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_constrained_trellis2.verify

/home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest discover \
  -s pose_constrained_trellis2/tests -p 'test_*.py' -v
```

See [`MIGRATION.md`](MIGRATION.md) for the precise old-to-new mapping and
[`VALIDATION.md`](VALIDATION.md) for the checks completed in this environment.
