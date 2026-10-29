# TRELLIS.2 geometry repair

Native official TRELLIS.2 Shape-512 generation and input-grounded local editing
of its final Shape SLat. Base weights and the geometry decoder are frozen.
There is no texture model, old30K checkpoint, or pose-adapter training in this
path. The upstream TRELLIS2 checkout remains unchanged.

Source lives here; assets, inputs, latents, OBJ meshes, figures, logs and all
runtime caches live under `/data/zjr/trellis2_geometry_repair`.

```bash
bash trellis2_geometry_repair/run.sh assets
bash trellis2_geometry_repair/run.sh prepare
bash trellis2_geometry_repair/run.sh tests
bash trellis2_geometry_repair/run.sh launch trellis2-repair-pilot-20261028 --gpu 6
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run.sh status'
```

See [PROTOCOL.md](PROTOCOL.md), [VALIDATION.md](VALIDATION.md), and [审计.md](审计.md). The initial pilot uses
four previously inspected development objects, two seeds, one reference image
for stock generation and posed multiview images for registration and repair.
It is a backend/repair pilot, not an independent generalization benchmark.

The completed pilot accepted no natural-input edits. The follow-up keeps those
results frozen and diagnoses depth-loss saturation and editing reachability:

```bash
bash trellis2_geometry_repair/run_diagnostic.sh tests
bash trellis2_geometry_repair/run_diagnostic.sh launch t2-diagnostic-20261028 --gpu 6
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run_diagnostic.sh status'
```

See [the four-arm diagnostic protocol](experiments/repair_diagnostic/PROTOCOL.md).
Input-derived latent/mesh editors are reported separately from GT-target
oracle diagnostics. Outputs live under
`/data/zjr/trellis2_geometry_repair/outputs/repair_diagnostic_20261028/`.

The next independent experiment uses frozen native Flow resampling, RGB-verified
evidence, connected 3D edit regions and optional local SS updates. Direct native
outputs and exterior-preserving FDG composites are explicitly separate:

```bash
bash trellis2_geometry_repair/run_local_regeneration.sh tests
bash trellis2_geometry_repair/run_local_regeneration.sh launch t2-local-v2-20261029 --gpu 4
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run_local_regeneration.sh status'
```

Read [the local regeneration protocol](experiments/local_regeneration/PROTOCOL.md).

The default local-regeneration profile is now `v2`. It uses voxel footprints,
RGB-verified support anchors and a fixed regional depth-scoring domain. V1's
protocol, code snapshots and partial results remain separate; see
[the v1 protocol](experiments/local_regeneration/PROTOCOL_V1.md) and audit §5.1.
V2 outputs are under
`/data/zjr/trellis2_geometry_repair/outputs/local_regeneration_v2_20261029`.
Its technical gates and eight-case development experiment are distinct from
the previous frozen native pilot and four-arm diagnostic.
To inspect the stopped v1 queue, set `T2_LOCAL_PROFILE=v1` before the status
command. Its original output directory is preserved unchanged apart from the
explicit stopped status and archived status history.

V2 closeout: all eight cases and full artifact checks completed at
2026-10-29T05:03:33+00:00. Read audit §5.3 and
`/data/zjr/trellis2_geometry_repair/outputs/local_regeneration_v2_20261029/FINAL_ANALYSIS.md`.
Adaptive delivered CD gain is only0.0402% with mixed F-score/novel-view results;
this validates the implementation, not reliable geometry repair.

The separate V3 implementation is described in
[guided_flow/README.md](experiments/guided_flow/README.md). It guides native
Flow clean-state predictions using continuous decoder fields and subdivision
logits, with input-only selections sealed before paired GT diagnostics.
Monitor it using `bash trellis2_geometry_repair/run_guided_flow.sh status`.
Its physical outputs are under
`/data/zjr/trellis2_geometry_repair/outputs/guided_flow_v3_fixed_20261029`.
