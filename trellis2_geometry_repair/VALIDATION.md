# Validated native512 backend

Validated on RTX3090, 2026-10-28. Current authoritative smoke completed at
12:54:59 UTC with decoder resolution explicitly set to512.

- Eight CPU protocol tests pass, including a regression that rejects the
  checkpoint's default256 decoder resolution and invalid32-grid latent coords.
- Official SS/Shape-512 generation runs. Learned checkpoint parameters match;
  omitted analytical RoPE buffers are recorded separately.
- Frozen-weight FlexGEMM compatibility: forward max difference0; input-gradient
  max difference0 against the same kernel with trainable weights. This is a
  process-local compatibility wrapper; no installed library file is edited.
- Native fixed-input FDG conversion repeated four times returns the same
  triangle set. Independent sorted-key and dense-grid integer lookups agree
  exactly with the original native converter. The diagnostic implementations
  are not used in the model path.
- Fixed-discrete-branch vertex parity is exact (max difference0). Finite latent
  gradient norm2.3673e-6. Directional finite differences have relative errors
  21.03%,15.60%,4.84% at epsilon.002,.01,.02 native-std units, respectively,
  satisfying the predeclared25% FP16 check.
- Native mesh is decoded at every recovery update.20-step generated-target
  recovery reduces fixed-ray depth MAE from0.0001358154 to0.0001104271
  (18.69%), and native geometry loss from0.00446045 to0.00368334 (17.42%).
  No scan GT or teacher-latent distance enters that objective.
- Smoke peak CUDA allocation6.61GiB; measured smoke time89.56seconds. These
  values describe the technical test, not full generation or natural-edit cost.
- SS coordinates and exterior feature rows stay fixed.

Receipts are under
`/data/zjr/trellis2_geometry_repair/outputs/native512_pilot_20261028/validation/`.

Earlier recovery failures used an erroneously unset decoder resolution256
with512 coordinates. They are invalid model-quality evidence, retained in
`validation/debug/` and `objects/005/42/invalid_decoder256/`. See the audit.
The authoritative decoder uses the original native conversion implementation.

The four-object/two-seed natural-input pilot runs separately. Its completed
results, candidates, fallbacks, GT metrics and figures are reported in the
output root's REPORT.md and summary.json. Passing this technical gate is not
an accuracy or generalization claim.

Natural-input pilot completed at 13:45:00 UTC on 2026-10-28: eight cases
(four development objects, two seeds) all have evaluation and comparison
figures. All 24 exported endpoint artifacts passed the fixed-SS and exact
outside-latent-row checks. No edit was accepted (0/8); delivered results equal
the input-aligned baseline. The final-step proposals average 0.218714% relative
CD improvement, but are not delivered method results. The 57.943888% average
raw-to-delivered CD improvement includes global registration and must not be
attributed to local editing. Native backend validation passed; natural repair
effectiveness remains unestablished.

The separate saturation/reachability diagnostic completed at 2026-10-29
00:43:09 UTC: 32/32 arms, 8/8 evaluated cases, 96 endpoint state checks.
Twenty-two CPU regressions pass. The old depth-field zero-gradient fraction
averages 33.07% on the sealed queries; the new uncapped robust objective gives
0%. Both actual-input arms accepted zero edits, so delivered gain remains zero.
Unfiltered mesh proposals can improve CD while producing spikes; rendered
comparisons were inspected and these are not counted as method successes.
GT-target arms and the subsequent fixed-SS support audit are diagnostics only.
See `outputs/repair_diagnostic_20261028/REPORT.md` and `FINAL_ANALYSIS.md`
under the physical project store, and the appended audit sections4.2/4.3.
# Local Flow regeneration v2 (2026-10-29)

The independent local-regeneration package passed 25 CPU regressions. The
real native GPU smoke completed at 04:24:49 UTC: exact captured-FDG/native
vertices and faces, exact empty-region mesh identity, exact exterior Shape
features, exact original SS-coordinate replay and exact exterior dense SS
latents. These are technical invariants, not evidence of improved quality.

005/42 completed the full input-only planning, SS/Shape proposals, decisions,
OBJ export, sealed GT evaluation and common-camera comparison. Independent
readback at 04:41:08 UTC validated all 8 candidate latents, 2 SS supports and
2 serialized composite fields for that case. The receipt is explicitly a
one-case partial audit; the full eight-case closeout remains the queue's job.
The first result has mixed metrics and small visible fragments; see audit §5.1.

```bash
bash trellis2_geometry_repair/run_local_regeneration.sh tests
bash trellis2_geometry_repair/run_local_regeneration.sh report --verify --index 5 --seed 42
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run_local_regeneration.sh status'
```

Physical outputs: `/data/zjr/trellis2_geometry_repair/outputs/local_regeneration_v2_20261029`.

## Local regeneration v2 completed

Completed 2026-10-29T05:03:33+00:00: 8/8 cases, 64 candidate latents, 16 SS supports and
13 serialized composites passed full readback. All eight figures were inspected.
The 13 scientific implementation files and v2 protocol match the launch snapshot.
Adaptive accepted3/8; delivered CD improved only0.0402% and metrics are mixed.
See audit §5.3 and the data-side FINAL_ANALYSIS.md; completion is a technical
result, not a claim of useful reconstruction improvement.
