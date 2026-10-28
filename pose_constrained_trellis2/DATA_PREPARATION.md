# Frozen ProObjaverse 30K preparation

This is the data path for migrating the existing frozen 30K experiment to
native TRELLIS.2 geometry supervision.  It preserves the old experiment's UID
partition and posed renders, but it does not reuse the old TRELLIS-v1 latent
labels.

## Current local boundary

The frozen source root
`/data/zjr/ProObjaverse-300K-ReconViaGen-30K` currently contains only:

- `renders_random_env`: posed RGBA tar archives used for DINOv3/K/T evidence;
- `lh-slats`: TRELLIS-v1 / ReconViaGen targets with 8 channels on a 64-grid.

It does not contain raw 3D assets, Flexible Dual Grid files, native
TRELLIS.2 Shape-512/1024 latents, or native TRELLIS.2 SS latents.  The old
`lh-slats` are explicitly rejected by `finalize` because TRELLIS.2 Shape-512
uses 32 channels on active 32-grid coordinates.

The released TRELLIS.2 `data_toolkit` also imports `datasets.ObjaverseXL`
without shipping that module.  The compatibility module under
`trellis2_toolkit_compat/` supplies the required current callback API without
modifying the sibling `TRELLIS2` checkout or depending on another project.

## 1. Export the frozen object identities

Export all 30,000 identities for native-target preparation.  Training itself
still uses the frozen `train` partition (29,872 requested objects); the other
128 identities remain isolated for decoder, bridge, and development audits.

```bash
cd /home/zjr/Tracker

/home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_constrained_trellis2.prepare_proobjaverse_cache instances \
  --split all \
  --output-dir /data/zjr/pose_t2_instances_30k
```

The resulting list is:

```text
/data/zjr/pose_t2_instances_30k/instances_all.txt
```

Its adjacent JSON file binds the list hash to the frozen selection and split
protocol.  Do not edit or regenerate the selection/protocol files.

## 2. Restore raw assets and build native TRELLIS.2 targets

Use an environment in which the official TRELLIS.2 toolkit and `o_voxel` are
installed.  The currently active `reconviagen` environment can validate this
project but does not presently provide `o_voxel`.

For the existing local environment, activate `trellis2`, install the official
data-toolkit dependencies, and then the two project additions before running
the commands below:

```bash
conda activate trellis2
cd /home/zjr/Tracker/TRELLIS2
bash data_toolkit/setup.sh
python -m pip install -r \
  /home/zjr/Tracker/pose_constrained_trellis2/requirements-project.txt
```

Set task-specific paths in each shell:

```bash
T2_REPO=/home/zjr/Tracker/TRELLIS2
T2_COMPAT=/home/zjr/Tracker/pose_constrained_trellis2/trellis2_toolkit_compat
T2_DATA_ROOT=/data/zjr/ProObjaverse-300K-TRELLIS2-native-30K
T2_INSTANCES=/data/zjr/pose_t2_instances_30k/instances_all.txt
```

Initialize an exact 30K metadata table.  `--source all` searches both released
ObjaverseXL catalogs and fails unless every requested SHA256 is resolved:

```bash
cd "$T2_REPO"
PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL \
  --source all \
  --metadata-instances "$T2_INSTANCES" \
  --root "$T2_DATA_ROOT"
```

For every distributed toolkit phase, launch ranks `0 ... W-1` with the same
`--world_size W`, then merge its records with `build_metadata.py` before the
next phase.  A single-rank form is shown here:

```bash
PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/download.py ObjaverseXL \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --rank 0 --world_size 1

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL --root "$T2_DATA_ROOT"

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/dump_mesh.py ObjaverseXL \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --rank 0 --world_size 1

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL --root "$T2_DATA_ROOT"

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/dual_grid.py ObjaverseXL \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --resolution 512,1024 \
  --rank 0 --world_size 1

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL --root "$T2_DATA_ROOT"
```

Encode Shape at both resolutions.  Shape-512 is the second-stage training
target; Shape-1024 supplies the high-resolution support from which the official
SS encoder constructs its target.

```bash
python data_toolkit/encode_shape_latent.py \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --resolution 512 \
  --rank 0 --world_size 1

python data_toolkit/encode_shape_latent.py \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --resolution 1024 \
  --rank 0 --world_size 1

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL --root "$T2_DATA_ROOT"

python data_toolkit/encode_ss_latent.py \
  --root "$T2_DATA_ROOT" \
  --instances "$T2_INSTANCES" \
  --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 \
  --resolution 64 \
  --rank 0 --world_size 1

PYTHONPATH="$T2_COMPAT:$T2_REPO/data_toolkit" python \
  data_toolkit/build_metadata.py ObjaverseXL --root "$T2_DATA_ROOT"
```

The two target directories consumed by this project are then:

```text
$T2_DATA_ROOT/ss_latents/ss_enc_conv3d_16l8_fp16_64
$T2_DATA_ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512
```

PBR dumping, texture voxelization, texture latent encoding, and TRELLIS.2
conditional rendering are intentionally not part of this geometry-only line.

## 3. Materialize posed DINOv3 evidence

The posed cache reads the already downloaded ProObjaverse render tars.  It
recovers only complete image/metadata pairs from trailing-damaged archives,
selects eight spherical-farthest views deterministically, applies the same
foreground crop to RGBA and K, converts C2W to W2C exactly once, and stores all
1029 DINOv3 tokens per view.  The source poses use OpenGL/Blender camera axes,
so each cache explicitly records `camera_forward_sign=-1` (after W2C
conversion) and `camera_image_y_sign=-1`; the latter maps camera `+Y` up to
image `+v` down.  Neither sign is inferred later inside the model.

Run one process per GPU.  For an eight-worker job, vary both visible GPU and
worker index from 0 through 7:

```bash
cd /home/zjr/Tracker
CUDA_VISIBLE_DEVICES=0 /home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_constrained_trellis2.prepare_proobjaverse_cache materialize \
  --split train \
  --output-dir /data/zjr/pose_t2_proobjaverse_train_cache \
  --device cuda:0 \
  --worker-index 0 --worker-count 8 \
  --record-source-errors
```

All workers must use identical arguments other than GPU and worker index.
Sidecars make the operation resumable and reject a changed preprocessing or
source identity.

## 4. Finalize a strict training manifest

The default is strict: zero source errors and zero native-target errors.  The
old 30K preparation observed 11 unusable training render archives, but this new
builder must report its own count.  If preprocessing also exposes unavailable
or failed raw meshes, inspect the toolkit records and then pass the exact,
audited counts explicitly; never guess a permissive bound.

```bash
/home/zjr/anaconda3/envs/reconviagen/bin/python \
  -m pose_constrained_trellis2.prepare_proobjaverse_cache finalize \
  --split train \
  --output-dir /data/zjr/pose_t2_proobjaverse_train_cache \
  --ss-target-dir \
    /data/zjr/ProObjaverse-300K-TRELLIS2-native-30K/ss_latents/ss_enc_conv3d_16l8_fp16_64 \
  --shape-target-dir \
    /data/zjr/ProObjaverse-300K-TRELLIS2-native-30K/shape_latents/shape_enc_next_dc_f16c32_fp16_512 \
  --max-source-errors 11 \
  --max-target-errors 0 \
  --deep-validate
```

`finalize` validates every target's native dimensions, records every excluded
UID and reason, applies the official Shape-512 limit of at most 8192 active
tokens per object (objects above it are excluded, never truncated), hashes
pose/SS/Shape artifacts, and loads the resulting manifest through both
training-stage dataset paths.  Include these over-limit objects in the exact
audited `--max-target-errors` count.  Only after this succeeds should
`scripts/gpu_smoke.sh` and the 30K training commands be run.
