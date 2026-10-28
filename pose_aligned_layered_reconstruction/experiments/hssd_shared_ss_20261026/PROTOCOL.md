# Matched SS training on shared-scene RGB observations

Previous cross and GT-support diagnostics show that accurate support permits
the existing frozen SLat to recover coarse geometry. The present experiment
tests whether matching the current input distribution and adding explicit RGB
geometry evidence improves predicted SS support. It does not train LaRI or SLat.

## Cohort, observations and controls

Retain chair i076, table i075, bench i080, stool i065 in the same old scene,
original object indices 0/5/6/7 and sampling seeds42/43/44. All use the existing
crop2072_visible DINO inputs and their original selected frames: 0/4/7/11 for
the first three, 0/11 for stool. Images entering DINO remain518 square.
The old frozen appearance_control predictions are12 archived references.
Two new arms each train200 steps and generate12 SS and12 meshes:

1. appearance_control: a new residual using the parent's appearance evidence.
2. geometry_observed: an equal-capacity residual using predicted surface,
   free-space and unknown-space evidence derived from those shared RGB frames.

Both share the complete old200-step appearance_control parent, including its
old residual and old no_layer base. The new residual has the same12-input MLP,
128 channels, LayerNorm, bias-free zero-initialized per-block projections,
initialization seed20261026 and trainable parameter count. Preserve every old
parent addition and negative-CFG path; initial outputs must exactly match it.
Freeze every parent parameter. No new parent/SLat/geometry-model fine-tuning.

Use the same four-object epoch permutation, stateless step noise/time, batch1,
200 exposures total (50/object), AdamW lr1e-4, betas(.9,.95), weight_decay.01,
warmup20, BF16, raw final weights, gradient clipping1. Train only the new
residual with the same SS latent velocity MSE. Do not add occupancy losses,
change targets, select checkpoints, or extend one arm's training on its result.
No claim of independent-seed training replication: seeds42/43/44 are sampling
repeats. Do not compare only a newly trained branch with the old frozen parent
to claim a geometry architecture contribution.

## RGB geometry and privilege boundary

Use the already local Stable-X/vggt-object-v0-1 checkpoint, full shared2072 RGB
resized to518 via the local official loader. Infer groups[0,4,7,11] and[0,11]
independently so stool receives no additional frames. This checkpoint is object
tuned, so scene-depth quality must be measured rather than assumed.

Convert predicted camera-z to known-camera world units using the median ratio
of known and predicted camera-center baselines. Record pairwise ratios, camera
rotations and center-alignment residuals. Never fit depth to GT depth, mesh,
occupancy, target extent beyond the already fixed canonical transform, or a GT
point prior. Scale cannot be defined for degenerate baselines; stop if invalid.

Raw confidence is1+exp(logit), not a probability. Use fixed per-view empirical
ranks, retaining rank>=.3. Combine with frustum checks, multiview evidence and
explicit conflict cues. No GT-dependent threshold choice or sample rejection.
Visible masks and known K/T/canonical frames remain the prior privileged input
contract. Thus RGB-predicted depth does not make this an RGB-only end-to-end
system. Same source frames are used in both arms; geometry additionally uses a
frozen VGGT and full-frame context, whereas parent appearance uses ROI DINO.
Equal trainable capacity does not mean equal total inference compute; this is
a comparison of conditioning pipelines, not an isolated fusion-layer ablation.

Project the canonical64-cube voxel centers into the selected known world
cameras and predicted depth rasters. Use only visible target masks, never
amodal masks or target-only depths. Delta is(camera z−predicted depth) in object
canonical units, tolerance2/64. Build soft visible-target surface evidence,
front-of-first-surface free evidence, behind-target/other-surface and unknown
evidence. Unknown is not empty; mask-outside pixels do not make all depths free.
Pool each4×4×4 cell into the corresponding16-cube SS token without changing xyz
order. Record all12 feature definitions, confidence/coverage and conflict stats.
The branch uses these fused features directly, independently of DINO view
weights. First version injects soft conditions, not hard occupancy carving.

The source caches have no model-readable GT depth. Existing evaluation_gt may
only be read by the separately named depth_audit.py after prediction/evidence
are frozen, solely to report depth errors. Its results do not select scale,
thresholds, checkpoints, masks or training examples. Complete GT-support mesh
results remain a clearly labelled external oracle reference, not a new method
arm or part of the36 primary logical cases.

## Gates, resumption and evaluation

Freeze new sources/protocol, immutable previous references, parent checkpoints,
depth weights/source code, RGB/masks/cameras and original target latent binding.
Keep all previous experiments unmodified. CPU preparation checks the original
SS latent/coordinates, image and SLat view selection, GT partitions and paired
table. Depth/evidence caches retain source contracts and exact hashes. No GPU
memory probe is allocated; explicitly select one GPU with a read-only query.

Before long training, require actual GPU checks of equal zero-init outputs vs
the full old parent, identical new parameter counts/hashes, unchanged frozen
parameters, finite new gradients, and two training steps in separate smoke
directories per arm. Require identical object order/time/noise between arms.
Replay oldtable2072/seed42 SS7751 coordinates and full SLat canonical/world mesh
and metrics exactly. Smoke checkpoints are not used as formal training starts.
Failures stop before formal stages. Persist the exact sampled SLat before
decoding; resume only if support, image, checkpoint, runtime and noise bindings
match. Preserve all failed/hard objects and all supports, including large ones.

After200 steps per arm, evaluate all24 new SS samples and complete meshes with
the unchanged frozen SLat30K,25steps,CFG5,interval[.5,1],rescale_t3,25positive/
19negative calls, original canonical-coordinate noise, LargeInputNative and
memory-efficient extraction. Include12 old reference meshes:36 primary logical
results. No refitting rotation/translation/scale or ICP. Known world placement
is not predicted scene layout. Score SS precision/IoU/count, CD,F@.01/.02,
visible/externally-occluded recall and free-space violation on fixed GT samples.
Report paired per-object/per-seed results, equal-object macro means and actual
meshes; do not gate completion on positive geometry gains.

Deliver12 native-camera sheets(GT/oldfrozen/newappearance/newgeometry) and9
known-layout world GLBs(three evaluated arms×three seeds). Both code/data and
training fairness must pass before interpretation. The assistant completes
short checks and launches; the user monitors the long queue. Do not expand
data or restart LaRI long training automatically. This four-object fit checks
learnability under shared inputs; a paper claim still needs unseen objects and
scenes, predicted camera/mask conditions and separate source/architecture ablations.
