# Geometry reconstruction and local repair — source snapshot

Current source snapshot of the TRELLIS.2 native geometry-repair work and its
30K/LaRI evaluation dependencies, exported on 2026-10-28. This is a dedicated
branch; the original Tracker worktree, index, remotes and other conversations'
branches are not modified.

- `trellis2_geometry_repair/`: official TRELLIS.2 SS + Shape-512 generation,
  native final Shape SLat editing, input-only registration/evidence/selection,
  OBJ export, evaluation, visual comparisons and protocol tests.
- `visual_evidence_probe/`: previous input-evidence and local-repair diagnostics,
  including the completed tenth experiment and its audit history.
- `pose_aligned_reconstruction/`: frozen SS30K/SLat30K training and inference.
- `pose_aligned_layered_reconstruction/`: the separate layered/LaRI experiments.
- `pose_constrained_trellis2/`: the separate pose-constrained training migration.
- Remaining directories preserve the existing local Python helper dependencies.

## Current TRELLIS.2 result

All 8 pilot cases completed at 2026-10-28 13:45 UTC. Eight CPU tests, native
vertex parity, frozen-weight gradient compatibility, numerical gradients,
controlled generated-target recovery and 24 endpoint artifact checks passed.
No natural-input edit was accepted (0/8); delivered geometry equals the
input-aligned baseline. This validates the native backend and complete pipeline,
not a natural-reconstruction accuracy improvement. The roughly 57.94% CD
improvement over unaligned raw outputs includes global registration and is not
local-edit improvement. Read `docs/trellis2_pilot_20261028/REPORT.md` and the
package's `审计.md` for the full distinction.

## Runtime setup and storage

This snapshot does not contain datasets, checkpoints, native model weights,
meshes, figures, caches, or the large Tracker history. Small result receipts are
materialized under `docs/`; audit symlinks are exported as real Markdown files.
`SOURCE_MANIFEST.json` identifies every exported source file by SHA-256.

Upstream code remains external. Clone the URLs and revisions recorded in
`docs/DEPENDENCIES.json` as sibling directories `TRELLIS2` and `ReconViaGen`.
Initialize their required submodules and install their native CUDA dependencies
according to the upstream instructions. The recorded local ReconViaGen patch is
included separately; no TRELLIS.2 upstream file was modified by this work.

Commands retain frozen machine-specific paths. On the original machine the
launchers use `/home/zjr/Tracker`, the Conda environments under
`/home/zjr/anaconda3/envs/`, and physical data under `/data/zjr/<project>/`.
A clone at another path needs explicit path mappings or launcher adjustments;
it must not silently reuse a different dataset or checkpoint. In particular,
`assets.py` expects the recorded model snapshots in the local Hugging Face
cache, and `prepare.py` reads the prior dense-evidence inputs. Those assets must
be provisioned separately before a fresh machine can run the pilot.

Original-machine commands:

```bash
bash trellis2_geometry_repair/run.sh tests
bash trellis2_geometry_repair/run.sh status
bash visual_evidence_probe/run_failure_repair.sh status
```

Every new dataset, output, temporary file and model/runtime cache belongs under
`/data/zjr/`; home/Tracker is reserved for source and small documentation.

## Follow-up diagnostic (2026-10-29)

The separate four-arm diagnostic is complete: 32/32 groups, 8/8 evaluated
cases, 96 endpoint checks and 22 passing CPU regressions. The uncapped robust
objective restores gradients for large depth residuals while retaining fixed
query domains, missing-ray penalties and exterior protection. OBJ surface
parity and the original float64 GT sampling contract are checked explicitly.

Both actual-input arms accepted 0/8 edits; delivered gain remains zero.
Unfiltered mesh proposals can reduce CD while creating spikes. GT-target
controls and the frozen-SS distance audit are diagnostics only. Read
[the final analysis](docs/repair_diagnostic_20261028/FINAL_ANALYSIS.md) and
[full report](docs/repair_diagnostic_20261028/REPORT.md). Source and commands
are in `trellis2_geometry_repair/experiments/repair_diagnostic/` and
`trellis2_geometry_repair/run_diagnostic.sh`.

## Native local/regional Flow regeneration (2026-10-29)

The new independent package performs input-RGB revalidation, retained global
registration, connected 3D regions, native masked Shape Flow resampling and
optional local SS regeneration. V2 uses voxel footprints, verified support
anchors and a fixed regional depth scoring domain. Exterior SS coordinates
and denormalized features are preserved exactly. Direct-native latents and
explicit FDG field composites are saved and labeled separately. The v1
protocol and results remain separate; thresholds were not changed.

25 CPU regressions, a fresh native GPU smoke, the full eight-case experiment,
and readback of 64 candidate latents, 16 supports and 13 exported composite
fields completed. All eight common-camera comparisons were inspected. The
scientific code and protocol match the launch snapshot; only orchestration,
reporting and validation were extended during the run.

Adaptive accepted3/8. Delivered CD improved only0.0402% against the same
registered base, while F-score@0.01 and the previous good-ray metric worsened.
This validates the implementation; it does not establish reliable geometry
repair. No acceptance thresholds were changed to improve the GT result.
Read `docs/local_regeneration_20261029/FINAL_ANALYSIS.md` and `REPORT.md` for
the completed outcome. Historical partial receipts remain explicitly labeled.

```bash
bash trellis2_geometry_repair/run_local_regeneration.sh tests
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run_local_regeneration.sh status'
```

Read `trellis2_geometry_repair/experiments/local_regeneration/PROTOCOL.md` and
audit sections5.1–5.3 for evidence, limitations, preserved failures and storage.

## Native geometry-guided Flow V3 (2026-10-29)

V3 updates predicted clean latents during the last three Euler steps using
native continuous surface fields, subdivision logits, input geometry and
preservation losses. Weights remain frozen. The same SS/noise prior controls
and a measured-cost prior bank accompany input-guided proposals. Method
choices are sealed before paired GT-domain/input-depth and GT-depth diagnostics.

All8 cases and full readback of112 latents/36 composites completed.22 CPU
regressions and native GPU parity/missing-child/zero-guide checks passed.
The initial variable-domain objective and the subsequent FP16 reduction bug
were audited and corrected before the final experiment; their artifacts stay
separate. Actual input accepted1/8: delivered CD gain0.1177% vs base, only
0.0167% vs budget-matched prior. Metrics remain mixed and numerically sensitive.
Read `docs/guided_flow_v3_20261029/FINAL_ANALYSIS.md` and the fixed-noise audit;
technical validation does not establish reliable repair.

```bash
bash trellis2_geometry_repair/run_guided_flow.sh tests
bash trellis2_geometry_repair/run_guided_flow.sh status
```

The original-machine launchers and inherited data/cache dependencies remain
explicit. Physical V3 outputs are under
`/data/zjr/trellis2_geometry_repair/outputs/guided_flow_v3_fixed_20261029`.

## Trainable pose-conditioned native T2 repair (2026-10-30)

`pose_conditioned_geometry_repair` freezes the official SS/Shape backbones,
trains multiview/reference-state adapters and attention LoRA, and learns a
damaged-to-clean Shape residual bridge. A frozen copy of the pose-conditioning
core avoids coupling to another active source project. No old30K weights or
distillation are used; texture is outside this experiment.

The initial Shape experiment failed and remains recorded. The residual bridge
passed32-object known-ROI overfit checks: matched-support CD improved12.30%, and
predicted-SS CD improved16.78% versus the actual damaged input. This establishes
training-set repairability, not held-out generalization or automatic localization.
Read the package audit and `docs/learned_geometry_repair_20261030` for controls,
the corrected physical-input baseline, validation and limitations.

```bash
bash pose_conditioned_geometry_repair/run.sh tests
bash pose_conditioned_geometry_repair/run.sh status
```

Original-machine paths/environment are explicit. Formal Train2000/Dev64 uses
data-validity screening before model training; large datasets, checkpoints,
meshes, images, caches and runtime logs are excluded from this source snapshot.
