# V3: geometry-guided native Shape Flow (2026-10-29)

Freeze V2 and all earlier experiments. All new tensors, OBJ meshes, plots,
logs, temporary files and caches are under `/data/zjr/trellis2_geometry_repair`.
This is a development mechanism experiment on the same four scanned-object
render sets005,006,010,011, seeds42,43; it is not independent generalization or
real-photo evidence. No model weight training, upstream edits, GT registration
or GT-chosen regions. Cases010 have global/structural errors and are reported
separately; cases011 have broad regional edits rather than tiny local edits.

## Fixed inputs and controls

Reuse V2's sealed input-only global transform, envelope, verified depth maps,
foreground masks and the `ss_t065_n101` support. The exact same SS coordinates,
noises101/202, original single-image context, normalization, 12 Euler calls and
exterior reference trajectory are used by prior/input/paired-input/oracle.
New coordinates start from full noise. The original aligned stock base and
V2 delivered adaptive result remain additional frozen references.

`prior`: two unguided native Flow proposals. First-case replay must match V2.
`input`: two proposals with geometry guidance on the last3 Euler calls.
`budget_prior`: input-selected prefix of an unguided bank, min2/max8 noises
101,202,303,404,505,606,707,808. Freeze its size before any GT read, using
measured sampling+native scoring time to cover input-arm cost. Discrete
overshoot and insufficient bank coverage are reported, not hidden; do not
claim exactly equal wall time or use cached zero runtime as the baseline cost.
Costs exclude common initialization and GT evaluation, which are separate.

## Guidance and missing geometry

Use each Euler prediction of the clean state. Clamp exterior clean rows to
the original clean reference before decoding. Backpropagate only through the
native Shape decoder, not through the 1.3B Flow network. Update editable clean
rows, convert the change back into an Euler velocity correction with the
upstream sigma/time convention, then continue the same trajectory. Exterior
noisy rows follow the original same-time path and final denormalized exterior
features are restored bit-for-bit. Flow weights are offloaded during decoder
backward to fit a3090; record full peak memory and actual transfer/guidance time.

At each of3 guidance times use one normalized gradient update, lr.08, per-value
cap.25std, backtracking fractions1,.5 based only on that arm's objective.
The decoder recomputes native discrete branches on every call. No stale sparse
cache, frozen GT topology, straight-through hard threshold or free vertex warp.

The gradient path bypasses final mesh extraction and retains native continuous
FDG fields and all four subdivision logits. Confirmed surface points supervise
the native child route with x+2y+4z channel ordering. This provides a gradient
at an existing parent whose desired child is currently absent. A missing
32-grid SS parent is explicitly unreachable under this fixed-support test;
no claim that child-logit guidance can create an absent coarse SS token.

Fine-field surface guidance uses a local soft minimum over nearest dual
vertices and the probability of at least one oriented edge crossing. The3
crossing logits are NOT treated as voxel occupancy. This is an approximate
continuous surface constraint, checked against the actual native mesh, not a
guarantee of valid new topology. Free-space samples are.02/.05 camera-z units
in front of confirmed surface rays; their fine-cell crossing probability is
penalized away from known target surface points.

Also optimize native rendered verified depth and observed silhouette. Protect
baseline geometry outside the envelope and input-consistent stable rays inside
it, plus sampled exterior continuous fields/subdivision probabilities. The
weights, query cap512,4096 field anchors and512 subdivision-parent anchors are
fixed in config.py before results. Losses use fixed denominators and do not
drop rays when a proposal loses the surface. A constant missing-face penalty
alone has zero geometry gradient; the continuous routing/surface terms supply
the additional path and are tested explicitly.

## Input decisions, diagnostics and artifacts

Only direct native meshes participate in V3 selection. Retain V2's frozen mask,
depth-gain, available heldout-RGB, exterior and topology gates. Save paired FDG
composites for selected endpoints as clearly labeled diagnostics, not as the
single-latent method output. Select accepted and lowest-input-cost unfiltered
candidate separately, and retain original base on rejection.

Seal all prior/budget/input decisions and budget allocation BEFORE loading GT.
Then build a diagnostic common query mask = input-confirmed rays intersecting
the GT surface. `paired_input` uses original input depths on this common mask;
`oracle` uses GT depths on exactly the same mask and the same preselected
pixel IDs. Both are explicitly GT-dependent diagnostics. Actual `input` keeps
its original full query mask. Same SS, ROI, protection, noise and sampler;
report oracle targets outside the envelope/support and never expand either
using GT. Diagnose evidence versus reachability without counting oracle as a
method result. Existing RGB masks remain input observations for every arm.

Evaluate after sealing with float64 GT surface sampling, unchanged per-case
global pose, CD/F-scores, the original eight novel views and fixed original
good/bad domains. Also report unclipped camera-z MAE on fixed GT/base common
rays plus lost-support fraction; frozen clipped silhouette/depth metrics remain
separate. Inspect all common-camera rendered comparisons and fragments/holes.
Count all fallbacks and costs. A lower training objective or technical smoke
passing does not establish successful reconstruction repair.

CPU tests precede native GPU field parity, missing-route gradients, zero-guide
identity, exterior invariance and a real nonzero geometry update on005/42.
Then validate the first complete method/paired-oracle case before continuing
the eight-case queue. Keep failures and source snapshots; create a distinct
experiment if scientific parameters must change after looking at results.
