# Native TRELLIS.2 repair: saturation and reachability diagnostic

Protocol fixed before this new run, 2026-10-28 UTC. Source and documentation stay
in Tracker; every runtime artifact is under
`/data/zjr/trellis2_geometry_repair/outputs/repair_diagnostic_20261028`.
The completed `native512_pilot_20261028` and its objective are retained unchanged.

## Cohort and questions

Reuse all four already-inspected development objects005,006,010,011, seeds42,43.
No regeneration, example removal, GT registration, reflected alignment, or
per-arm similarity fitting. Read the exact original latent, fit and sealed
input-only region/queries; bind the small input artifacts by SHA-256.
These are scanned assets with rendered RGB inputs, not real photographs.

The original depth objective has zero derivative for absolute residuals larger
than0.1. Measure actual unclipped residuals, saturation fraction and pixel-depth
gradients, not just a nonzero aggregate latent gradient. Large teacher residuals
may also be evidence errors; gradient restoration alone does not make them true.

## Four arms

1. `latent_input`: original native final Shape SLat rows and input-derived
   DA3/NCC targets, with uncapped robust depth loss.
2. `mesh_input`: smooth local graph deformation of the same initial native
   mesh, with the same input-derived targets.
3. `latent_oracle`: the same latent editor, supplied GT camera-z only as an
   explicitly labeled diagnostic upper-bound target.
4. `mesh_oracle`: graph deformation with GT target, diagnostic only.

Both input-only arms must seal their final input decisions before oracle mesh
loading for that case. Their training and selection functions receive no GT
geometry, masks, cameras, errors or losses. Oracle targets use the same
INPUT-defined region and query proposals; GT-missing queries have no oracle
depth target and their counts are reported. GT visibility never filters or
reweights the input arms. No oracle score is mixed into actual method results.

Train views and heldout split remain the original sealed split. One reference
view was used in stock generation; the heldout view is for editing, not a claim
of a novel reconstruction input. Eight original novel evaluation cameras,
global transform and GT scan remain identical across arms.

## Objective and editors

Depth uses Charbonnier residual/0.03, epsilon0.01, with nonzero large-error tails.
The denominator and query masks are fixed. Each missing ray pays a fixed cost
based on its ORIGINAL residual plus0.05, floored at0.1; removing a baseline ray
cannot lower its depth loss. Coverage remains differentiable. Acceptance also
requires no more than0.5% loss of fixed local target support. Existing mask,
normal, coverage and outside protection definitions remain unchanged. Evaluation
keeps the old fixed-domain capped metric for comparability; raw residuals are
additionally reported for training/evidence audits.

Latent:40 Adam updates, lr0.02, original active rows, cap0.25 native std,
prior1e-4, loss scale1024. Native512 decoder and frozen weights; decode actual
topology each step. SS and exterior latent rows stay exact. The kernel wrapper
and512 guards remain the already-validated native implementation.

Mesh:40 Adam updates, lr0.02;96 deterministic surface anchors,4-neighbor smooth
interpolation,0.015 world-unit boundary taper, per-axis translation cap0.15,
parameter prior1e-4 and neighbor smoothness1e-3. Only vertices near the same
input-visible region can move; other vertices stay EXACT. Faces stay fixed.
These editors have different parameterizations and constraints. Their contrast
diagnoses the current editing setup, not an intrinsic impossibility theorem
about the entire learned representation.

Checkpoints0,10,20,40 are chosen by each arm's own training loss and the
predeclared input safety gates. Record unfiltered final-step proposals as well
as selected candidates and delivered fallbacks. GT is never used to choose an
input-arm checkpoint. Original native OBJ fallback is reused; store small
checkpoint states and export only selected/final meshes to reduce data I/O.

## Reading the outcome

Primary actual-method comparison: input arms' delivered outputs versus the
SAME input-aligned base, all eight cases including fallbacks. Also report
candidate/final-step CD,F-scores,fixed-domain bad/good/new-view errors, local
target fit, outside drift, missing support, timing and VRAM. Timing includes
optimization/checkpoint/export and excludes model/input loading and final GT
evaluation. Oracle model VRAM can include the rendered GT target, so raw GPU
memory numbers are not strict equal-input inference benchmarks.

For evidence diagnosis, compare mixed DA3/NCC targets and the original mesh
against GT on the fixed input-supported rays AFTER sealing input decisions;
report common-support fraction, raw MAE, signed bias and ray improvement rates.
Never fit teacher scale or registration to GT and pass that back to input arms.

If oracle latent fitting is substantially stronger than input fitting, inspect
evidence. If oracle mesh can fit but latent cannot, inspect the current native
latent optimization/support constraints; local Flow resampling or SS support
editing are subsequent options. If both fail, inspect registration, region and
protection objectives. This is a development diagnosis; no generalization or
independent held-out success is claimed from these already-viewed objects.
