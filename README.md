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
