# Round ten: repair natural generation errors

Frozen on 2026-10-28 before inspecting new-cohort outputs.

Rounds 1--9 are retained as development evidence. They are not nine trained
model versions. No old result is overwritten or rerun. The complete method for
this round is the existing round-seven input-only dense-evidence editor with
the observed-mask acceptance policy. This is a task-focused independent-cohort
evaluation, not a claim of a newly trained refiner or a new depth algorithm.

Select 16 categories by a fixed SHA256 order (seed 20261028), then the first
eligible UID in a separately hashed order per category from the existing
Omni200 manifest. Exclude every Omni UID in the previous round-seven registry
(all eight development and sixteen earlier validation objects). Inputs are the
existing four posed rendered views per object; seeds are 42 and 43. Do not call
these real photographs or claim foundation-model pretraining disjointness.
The cohort is new to this local-repair development, not necessarily to every
other project in Tracker. Failure status never selects or removes a run.

Three delivered assets per case are scored: raw30K, input-aligned30K, and
input-aligned30K plus the complete correction procedure. The primary repair
comparison shares the exact same input-derived global transform. Raw30K gives
the end-to-end reference and prevents registration gains being mislabeled as
local repair. Candidate and final80-step outputs are secondary diagnostics.

All SS coordinates, weights, decoder and outside latent rows remain fixed.
The inherited 80-update, 0.5std-cap, learning-rate .03 editor and final gate are
unchanged. NCC anchors and posed DA3 estimates come only from the three editing
input views. The fourth view is excluded from the edit targets and gradients,
but was used by base generation/global alignment and later acceptance. Eight
additional evaluation cameras are genuinely absent from generation/editing.
GT provides no pixel-support pruning, edit region, target, transformation,
stopping or checkpoint selection. The evaluator is a separate process and
requires both seeds' sealed input_complete records before loading scan meshes.

Before editing, save an input suspicion score: 0.5 times mean silhouette
disagreement plus 0.5 times min(median local trusted-depth discrepancy / .03,1).
When no depth evidence exists, record that explicitly and use silhouette only.
This is an analysis/ranking score, not a newly validated correctness probability
or an extra deployment gate. The inherited pipeline may keep a case unchanged.

After evaluation, report all16 objects plus predeclared upper/lower halves by
object-mean aligned-base CD, and separate upper/lower halves by input suspicion.
The GT quality strata are evaluation-only, not automatic case selection. The
lower-error half is a relative control, not guaranteed error-free geometry.
Seed observations are averaged within an object before bootstrap statistics.
The initial-error versus gain plot is descriptive: both axes share a baseline
term and correlation alone is not causal evidence of selective correction.

Also freeze per-pixel error masks on the aligned baseline at eight novel
cameras: error >= 1/256, on the union of baseline and GT foreground. Measure
improvement on those originally bad pixels, harm on originally good pixels,
localization coverage/precision, and selected-bad-region error. Missing rays
pay the inherited0.1 penalty. These evaluation masks never reach the optimizer.
Report continuous CD/F-score/depth effects, acceptance/no-op rate, topology,
outside drift and timings. A descriptive material-repair count requires both
>=1% global CD reduction and >=5% originally-bad-pixel error reduction; it is
not a statistical significance test. Report every fallback and failed process.

First run one new object x both seeds as an end-to-end technical smoke. Check
decoder parity, finite metrics, fixed SS/outside rows, expected files, disjoint
UIDs and input/GT separation. No numerical-gain threshold controls proceeding.
On technical success, automatically run the remaining15 objects in background.
CUDA synchronous execution is fixed for this round after round-nine runtime
failures. Execution timing is descriptive, with no cross-GPU speedup claim.

TRELLIS.2 and real-photo benchmarks are subsequent cohorts. This round must not
claim results against those baselines. Historical reanalysis remains explicitly
development analysis and is stored under this new output root.

All runtime data, renders, caches and temporary files live under
/data/zjr/visual_evidence_probe. Source and small documentation stay in Tracker.
