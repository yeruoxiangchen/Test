# Round nine: matched-support evidence accuracy x editor diagnostic

2026-10-28. Four inspected development objects (005,006,010,011), two seeds,
four arms: estimated_free, estimated_anchored, oracle_free, oracle_anchored.
These cases were deliberately chosen to cover evidence errors and potential
overprotection. They are not a new validation set.

ALL arms are privileged diagnostics: the common support is the intersection
of the round-eight UNWEIGHTED NCC+DA3 valid pixels, original input-derived local
region, training views, and actual positive finite GT input depth intersections.
The estimated arms also use this GT-pruned mask and are NOT deployment scores.
All four arms are rerun; old round-seven/eight scores cannot be reused as the
estimated-arm result after changing the supervised support.

The only source change is camera-z target value on identical pixels. No GT
silhouettes or GT normals are added: observed input silhouettes and original
exterior protection remain identical. The preparation process exports input
depth only; novel evaluation depth is never written to optimizer targets.

30K and decoder weights, original SS, editable rows, original world transform,
input cameras, train/heldout split, region, 80 attempted updates, .5std feature
bound, lr.03, static loss scale1024 and final input gate stay fixed. All4 arms
share the estimated-target INITIAL loss normalization within each case, so
changing depth accuracy does not independently rescale the optimizer objective.
The anchored editor exactly retains round-eight anchors, exterior normals and
five-step rollback with corrected deep-copied Adam states. No new tuning.

Primary endpoints are own-training-selected CANDIDATE and fixed80-step PROPOSAL
on eight novel evaluation cameras and global mesh metrics. The oracle training
selection is privileged. Input-gate accepted results are secondary and cannot
by themselves establish an editor capacity limit. Report no-support cases,
removed GT-missing pixels, initial depth quality, clipping saturation, gradient
health, rollbacks, actual time and outside drift. Finite budget and the clipped
loss prevent interpreting an oracle failure as a universal impossibility.

Differences to inspect: oracle-estimated for each editor (evidence accuracy),
anchored-free under each depth source (constraint effect), and their interaction.
Average seeds within the four objects; object bootstrap is descriptive only.
All outputs, caches and temporary data stay under /data/zjr/visual_evidence_probe;
source code and small documentation stay in Tracker. Older results remain intact.

Numerical control added after the first smoke, before remaining results: initial
losses replayed exactly across old/new runs, but small gradient differences
produced different80-step latents. All four primary arms run fresh on the same
GPU/process. Before closeout, repeat estimated_free twice on005/42 in one process,
with identical parameters/support/noise. These2 auxiliary runs are not method
arms or independent samples; report endpoint variation and do not interpret
tiny gains below observed variation as established mechanism improvements.

Closeout execution recovery,2026-10-28: the32 main arms, metrics and figures
finished successfully. The first extra repeat audit failed in the stock
FlexiCubes regular-volume indexing (16,777,216-element mask vs16,777,215 indices).
Keep the failed log and retry only the extra controls in a fresh process with
CUDA_LAUNCH_BLOCKING=1. No model, targets or scientific hyperparameters change;
record the execution mode. Repeat1/2 compare the same synchronous mode, while
original-versus-repeat also includes mode/run differences and is not a pure
random-noise measurement. Do not rerun or overwrite primary method results.
