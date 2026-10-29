# Native local/regional regeneration v2, 2026-10-29

V1 is retained in `PROTOCOL_V1.md` and in its separate output directory
`outputs/local_regeneration_20261029`. Its first complete case showed that
center-only foreground checks discard boundary surface cells, and that depth
scoring outside the editable volume can hide the local effect. V2 fixes those
input-side implementation issues, in `outputs/local_regeneration_v2_20261029`.
This remains development on already-inspected objects, not a fresh test set.
The original acceptance thresholds are unchanged; v1 results are not re-scored.

This is a new protocol. The native pilot and completed four-arm diagnostic stay
frozen. Source is under Tracker; datasets, model assets, temporary files, model
caches, tensors, OBJ meshes, figures and logs stay under
`/data/zjr/trellis2_geometry_repair`. No training or upstream-file changes.

## Scope and comparisons

Reuse all four already-inspected development objects005,006,010,011, seeds42,43.
These are posed renders of scanned objects, not real photographs or held-out
generalization data. Stock generation used only reference image0. Three input
views provide editing evidence; the original fourth view is RGB validation.
All four masks can participate in global registration. No claim that validation
is a never-seen reconstruction input. GT is read only after input decisions.

Compare old input-aligned stock output, new registration-only stock output,
`shape` (fixed-SS Shape Flow resampling) and `adaptive` (all Shape proposals plus
local SS/Shape proposals when reliable support-update evidence exists).
Adaptive is the primary pipeline; it can choose an ordinary Shape proposal.
No separate per-candidate or GT alignment. Registration and local-edit gains
are reported separately, as is the full change relative to the previous base.

## Input evidence and global pose

Recompute NCC from the current mesh. Cached posed DA3 is checked against the
same provided cameras, then re-verified by RGB patch texture, NCC improvement,
foreground reprojection and counterpart depth agreement. The old mixed depth
targets are not used directly. No GT scale/pose calibration or GT region labels.
Defaults: patch5; texture std>.02, cost<=.30, improvement>=.015, other-view
camera-z agreement<=.02. Narrow-range NCC also requires texture std>.02.
Heldout RGB does not enter these target proposals.

Retain the original proper global similarity. Propose24 proper orientation
permutations and refine two mask-ranked alternatives for30 Adam updates.
Fit mask cost plus0.05 verified robust-depth cost. Keep a new global fit only
when its objective improves, each observed mask loses<=.005IoU, and available
fixed-support heldout RGB is non-regressing. No reflection, ICP-to-GT or mesh
deformation in this stage. Recompute evidence after the selected global fit.

## Continuous 3D edit regions

Build32-grid error weights from verified depth discrepancies>=.005, observed
foreground violations, and mask deficits consistent with the multiview visual
hull near existing support. Connect source/target ray endpoints. Select a
connected high-score component, cap the core at1024cells, and add a two-cell
blend/protection buffer. Record active fractions and label broad edits
`regional`; a small token count is not a geometric locality guarantee.
Visual hull constrains free space; it is not directly filled as surface support.
V2 checks projected voxel footprints against each foreground mask rather than
requiring the voxel center to be foreground. At least5% footprint deficit is
needed to label a missing-mask cell. When at least16 verified discrepant rays
exist, select the connected edit region from their source/target depth tubes;
otherwise use silhouette evidence. This prevents dense mask pixels from
overriding the independently verified geometry. A verified support anchor must
have at least3 confirmed target rays in its voxel and satisfy the footprint
foreground constraints in all training views.

## Native generation

Use original official SS/Shape Flow weights, original single-image DINOv3
context, native normalization and Euler sample_once.12 steps, original CFG and
time rescaling. Partial Shape starts at0.35 or0.75, two fixed proposal noises.
At every step the exterior follows the ORIGINAL noisy path at the SAME time;
terminal denormalized exterior features are copied bit-for-bit from the base.
Each sparse evaluation has a fresh cache. Starting a partial trajectory at1
would be an error and is covered by tests.

When support updates are requested, replay original dense SS and verify its
decoded32-grid coordinates exactly. Locally re-noise dense SS at0.65 or1.0,
preserving exterior dense values. Six bounded differentiable SS-decoder
occupancy updates use only verified positives, visual-hull free space and
observed missing-mask coverage; no hard-threshold gradient or GT supervision.
Only core SS coordinates may change. New support must satisfy input foreground
constraints. Retain original buffer/exterior support. Then regenerate local
Shape features from full noise (new coordinates have no observed clean latent),
with two proposal noises. Original exterior coordinates/features stay exact.
V2 also unions the verified anchor cells into the core support after SS
thresholding, recording requested/before/after coverage and all added/removed
cells. This is explicitly native SS resampling plus input-derived anchors;
the resulting support is not claimed to be the unmodified native SS decoder
output. No anchor is created outside the core, and the exterior support remains
exactly the original set. Soft occupancy optimization alone did not guarantee
the verified cells in v1.

## Explicit geometry preservation

Keep direct native candidate latents and scores. Also create an explicitly
labeled FDG composite: original512-grid decoded cells/7channels outside the
3D envelope, new native cells in the core, and blended existing fields in the
buffer. Reuse the original native FDG converter. This composite is NOT the
direct decoder output of a single saved latent. Save its field state and the
paired direct-native OBJ for selected proposals. Never claim latent-only
reproducibility for a field-composite OBJ.

This protects decoded exterior fields by construction, beyond merely clamping
latent rows. Validate object-space exterior preservation away from the edit
collar in rendered views; allow physically valid occlusion by an edited region.
No free vertex displacement, mesh repair generator, or watertightness requirement.

## Decisions and evaluation

Select using input mask/depth cost, fixed-support heldout RGB non-regression,
3D exterior checks, component/area sanity. Require at least.001 mask-cost gain
or5% verified-depth gain; observed IoU loss<=.002 in EACH input. Training cost
must not regress. Exterior mean<=.001,p95<=.004,lost support<=.001. Composite
exterior raw fields must be exact. Additional small-component area<=.003,
surface-area ratio[.65,1.35], large components<=base+1. Keep the original if no
proposal passes. These are new pre-result gates, not re-scored old outcomes.
V2 freezes the depth scoring domain before proposals: keep a verified ray if
its baseline surface point or target point is in the input-defined edit
envelope. Global registration still uses all verified input targets. Keep
global mask checks, heldout RGB and exterior protection unchanged. Report both
global and local verified-ray counts; candidate-dependent visibility cannot
select the depth evaluation domain.

GT CD/F-scores and the same eight novel cameras are evaluation-only. GT scan
sampling remains float64 before rasterization conversion. Report all fallbacks,
candidate/delivered outputs, global versus local effects, scope, time and peak
VRAM. Inspect visuals; scalar gains alone do not prove useful repair.

First validate CPU invariants and native GPU replay/field-conversion/empty-edit
parity on005/42, then the first complete case and the full eight-case queue.
Success is not assumed; no acceptance threshold is tuned to the current GT.
