# Validation record

## 2026-10-28: actual HSSD Native12 CUDA execution and mesh review

The earlier environment boundary below is historical. HSSD Native12 has now
completed official native Shape512/1024 and SS encoding, Shape512 decoding,
DINOv3 conditioning, two-step SS/Shape training plus resume to step 3, and full
12/12-step sampling on the first training object. All 12 codec meshes were
rendered as actual triangles and reviewed at fixed cameras without ICP.
Ninety stage receipts and 36 manifest file hashes match the frozen runtime.
All SS roundtrip IoUs are 1; mean Shape codec F@0.02 is 0.994552. Independent
20k-point GT self-sampling gives 0.994504, documenting finite-sampling error.
Both stages' raw/EMA states are finite; all 370 trainable tensors changed from
step 2 to 3. Initial raw and anchored stock differences are 0 on tested samples.

The generated chair after only three steps lacks most legs and does not match
the target proportions/frame; the codec chair preserves the major structure.
This qualifies execution, not successful reconstruction or a relation model.
GT cameras/object transforms remain inputs. No formal baseline or relation
training was launched by this review. See the [dated review and mesh figures](experiments/hssd_scene_20261028/REVIEW_20261028.md).

## Historical CPU-only record

Date: 2026-09-04 UTC

## Completed locally

- All package files compile under the `reconviagen` Python environment.
- All train, inference, preprocessing, and verification command-line entry
  points construct successfully.
- The source-independence AST audit found zero imports from the old model
  packages, GenRecon, or the former shared adapter modules.
- The locally cached official TRELLIS.2 SS and Shape-512 JSON configurations
  match the frozen resolution/channel/width/30-block/RoPE contracts.
- Safetensors metadata confirms the LoRA attention surface in every block of
  both official 1.3B Flows.
- Both complete official 1.3B Flow checkpoints were loaded on CPU and wrapped
  by the production builder.  SS and Shape each report 30 condition blocks,
  300 LoRA tensors, 70 condition tensors, and 60,295,169 trainable parameters;
  the trainable-surface whitelist passed for both.
- The locally cached SS decoder is the expected 8-channel decoder and the
  Shape decoder is the 32-channel `FlexiDualGridVaeDecoder`.
- Eighteen CPU tests pass, covering camera projection (including explicit
  OpenCV/OpenGL vertical-axis behavior), DINO prefix/patch token
  handling, view-order semantics, permutation-invariant aggregation, exact
  zero initialization, live per-block gradients for dense and sparse Flow
  adapters, inference pose-condition caching and stock-forward elimination,
  mask-crop/K consistency, strict native target loading and the official
  Shape-512 8192-token filter, SS-to-Shape support
  downsampling, inference view parsing, the frozen 30K selection/protocol and
  tar camera contract, per-rank checkpoint RNG state, hard-link aliases,
  exact toolkit metadata filtering, and safe ZIP/no-file toolkit callbacks.
- The public TRELLIS.2 data-toolkit gap was reproduced: its scripts import
  `datasets.ObjaverseXL`, while that module is absent from the released tree.
  The independent compatibility adapter constructs in this environment,
  exposes the callback required by `dual_grid.py`, and leaves TRELLIS2
  untouched.
- The exact frozen source was audited as 30,000 identities partitioned into
  decoder-audit 32, predicted-support bridge 32, development 64, and training
  29,872. A real source tar produced the same eight deterministic view IDs as
  the previous implementation, with C2W-to-W2C and forward-axis checks passing.
- A 16-object real ProObjaverse render/camera audit exposed an inherited
  vertical-axis omission in the old projection path.  Using old sparse support
  coordinates only as a read-only geometric probe, all 16 objects improved:
  mean weighted RGBA-silhouette hit rose from 0.6072 to 0.9162 and the median
  rose from 0.6317 to 0.9238 with the explicit OpenGL
  `camera_image_y_sign=-1`.  The most diagnostic 900-point object rose from
  0.1098 to 0.8137.  The migration records this sign in every ProObjaverse pose
  payload; old SLat features are not used as T2 supervision.
- The local 118 GB source root contains posed renders and old TRELLIS-v1
  `lh-slats`; native TRELLIS.2 SS and Shape targets are absent. This is
  reported as an execution boundary, not silently treated as ready.

Commands and results:

```text
python -m unittest discover -s pose_constrained_trellis2/tests -p 'test_*.py' -v
Ran 18 tests
OK

python -m pose_constrained_trellis2.verify
passed: true
forbidden runtime imports: 0
official SS LoRA-covered blocks: 30
official Shape-512 LoRA-covered blocks: 30
texture models checked or loaded: false
TRELLIS.2 toolkit compatibility adapter: PASS
native TRELLIS.2 targets ready: false

official SS production builder: 60,295,169 trainable parameters, PASS
official Shape-512 production builder: 60,295,169 trainable parameters, PASS
```

## Environment boundary

This Codex runtime has no usable NVIDIA driver, so it cannot execute an actual
1.3B BF16 CUDA Flow, official sparse CUDA kernels, a training optimizer step,
or a decoded Flexible Dual Grid mesh.  The GPU acceptance test therefore
remains intentionally unclaimed.

The existing `trellis2` Conda environment contains `o_voxel`, `cumesh`,
`flex_gemm`, and FlashAttention, but currently lacks `peft` and `objaverse`;
the existing `reconviagen` environment has the latter packages but lacks
`o_voxel`.  `requirements-project.txt` records the two additions needed on top
of the official TRELLIS.2 environment.  No environment was mutated during
this migration.

On a CUDA host, run `scripts/gpu_smoke.sh MANIFEST OUTPUT_ROOT`.  Its acceptance
sequence is:

1. Build a one-object native T2 cache with two or more posed views.
2. Train SS for two optimizer steps and verify finite loss, all 30 condition
   projections covered, a valid checkpoint, and resume to step three.
3. Repeat for Shape using native Shape-512 targets.
4. Run inference with one sampling step per stage, then the official 12/12
   sampler settings, and require a non-empty finite PLY plus JSON report.
5. Run a small held-out stock-versus-adapted comparison before scaling either
   stage to 30K.

Nothing in this record claims that a new 30K TRELLIS.2 model has already been
trained; it records that the independent migration code and its CPU-verifiable
contracts are ready for that GPU gate.
