# Fifth probe: editor objective and sparse support

2026-10-26. No model weights are trained. All runtime I/O is on /data/zjr.
The eight previously inspected Omni objects and two saved seeds are exploratory
development cases, not a fresh held-out benchmark or real-camera evaluation.

First check full mesh forward parity and latent backward on the actual GPU.
The stock complete SDF, deformation, cube weights and color path is retained.
Only an equivalent out-of-place SDF bias and channelwise mean with analytic
backward are introduced in a reversible process-local backend. Forward inference
uses the historical FixedNative policy; backward restores the complete native
GEMM tuner. Do not globally disable deterministic Torch operations.

Recovery: index4 asparagus and index6 carrot, seeds42/43. Construct a fixed,
small, spatially smooth perturbation of selected latent rows. Decode it to a
target mesh, then hide its latent values from the optimizer. Both objective
arms start at the same original latent, same support, same editable rows and
same feature bounds/budget. Compare corner-SDF proxy to rendered full decoded
depth/mask/normal with region-outside protection. Target is representable by
construction. Report actual target-mesh geometry improvement, not proxy loss
alone. A 50% rendered-depth reduction in at least3/4 cases is a development gate,
not proof of generalization. Keep failures and target perturbation magnitude.

Factorial: all8 objects x2 seeds x(originalSS/expandedSS)x(proxy/fullmesh).
Use the same input-only automatic constraint pool, confidence ranking and
at most512 points across treatments. The original global fit is fixed; no GT
alignment, ICP or per-arm renormalization. SS candidates are input target voxels
and short source-to-target paths, bounded by min(128, floor(.2*Nold)). Targets
outside the native cube are reported, not silently clipped into it.

New-row initialization is deterministic neighboring-original-feature
interpolation. Store raw expanded decode, then at most20 steps updating only
new rows to reproduce the original mesh. This initialization uses neither
correction targets nor scan GT. Share the same preserved initialization between
both objective arms. Report original->raw->preserved drift. Initialization is
not assumed neutral; failed drift gates are retained as init_failure and are
not evidence against SS expansion. Report final metrics against both original
and each arm's step0. Drift gates: mask xor/union<=.005; visible depth MAE<=.25/256,
p95<=1/256; normal mean angle<=2degrees. Exterior geometry drift is reported.

All editing rasters and fresh matching use the existing518 crop cameras at256
pixels, with the same pixel-center convention. Fullmesh local supervision uses only sparse input-proposed target camera-z;
unknown local normals are not invented. Baseline exterior depth/normals and
input silhouettes protect the rest. The controlled recovery has full target
normals. Missing predicted coverage pays a fixed penalty with a fixed domain.

Choose edit-heldout views from current mesh visibility of the intended region,
never a fixed fourth view. Omit that view's proposal anchors from optimization.
Validation must use fixed local pixel support, with a fixed missing penalty.
Insufficient support is unable, not successful or failed correction. Generation
and the old global fit used all4 views, so this is only edit-estimation holdout.

Evaluate all states, including no-constraint, unable, init_failure and failed
decoding. GT appears only after input-only checkpoint selection. Record surface
CD/F-score, visible depth, actual changed area, outside drift, components,
nonmanifold/boundary edges, time and peak memory. Compare seeds within objects;
bootstrap objects, not pixels/seeds. Do not claim SS alone causes failure.
