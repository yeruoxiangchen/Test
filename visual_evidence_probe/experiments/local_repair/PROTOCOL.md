# Fourth probe: can observed evidence improve local geometry?

2026-10-26. This is an exploratory continuation of the inspected round3 cohort,
not a fresh held-out benchmark and not an evaluation of a newly trained model.

Use all eight Omni scanned objects and both saved seeds. Inputs are four rendered
RGB observations with masks and calibrated cameras. Reuse each frozen SS30K/SLat30K
mesh and its existing input-mask global fit, identically across all arms. Never
add per-arm ICP, GT alignment, scale correction, or favorable-case filtering.

Views 0,1,2 estimate corrections. View3 is held out only from local correction
estimation, and selects an amplitude using a fixed-pixel-denominator patch NCC
objective and per-view silhouette regression gate. Generation and the prior
global similarity fit already used all four views; view3 is not an unseen input.

Estimate local camera-z corrections with a fixed +/-0.06, 0.005-spaced depth
sweep. Use 5x5 normalized grayscale patch agreement, foreground erosion,
baseline angles, original-mesh visibility, texture support, two supporting
training views, and per-pair depth agreement. Reject uncertain, ambiguous,
search-boundary and zero-depth proposals. No GT is accepted by this module.

Select 20% of eligible visible triangle area using input-only cross-view RGB
residuals, a seeded random ordering, or GT visible depth-error ranking as an
explicit oracle-region diagnostic. All three consume exactly the same input
depth proposals and bounded fixed-topology least-squares mesh deformation.
An extra oracle-direction diagnostic uses GT depth corrections to isolate the
solver's ability; it is never reported as an input-only method. Record the actual
triangle area because a final triangle can slightly exceed the target budget.

Use amplitudes 0,0.25,0.5,1. Keep amplitude zero unless held-out input NCC improves
by >=0.0001 with no view's silhouette IoU dropping by more than0.005. Keep the
unconditional amplitude1 proposal for diagnosis, including rejected attempts.
All controls get identical rules. Save zero-edit and failed cases.

Evaluate deterministic 50K surface CD/F-score, fixed original-frame visible
depth error, originally erroneous and originally correct visible regions,
silhouette coverage, actual changed area, proposal support and time. Aggregate
seeds within objects and bootstrap objects only. Perfect-geometry inputs must
also be checked for false edits. An analytic textured-plane test must establish
depth sign/scale and an analytic mesh test must establish bounded local updates.

Only after this geometry closure, test the same local constraints against a
frozen decoder by optimizing sparse SLat features in selected spatial support.
Recreate the original seed latent, establish decoded-mesh parity, preserve
original SS coordinates and original global fit, and verify frozen token rows
and region-outside surface leakage. Record any memory/autograd limitation.
This stage targets existing surfaces; adding missing sparse support is outside
this round. No weights are trained. All runtime I/O is under /data/zjr.
