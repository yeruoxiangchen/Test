# Native TRELLIS.2 repair pilot, 2026-10-28

Use stock official SS and Shape-512 Flows and the official Flexible Dual Grid
decoder. Save denormalized Shape features, native32-grid coordinates, mean/std,
reference view, seed and native OBJ before any registration. Every decode
constructs a fresh SparseTensor; no encoded-GT or stale subdivision cache is
allowed. Load and release image encoder, SS, Shape, decoder in separate stages.

Inputs: the four already-inspected Omni development objects005,006,010,011,
seeds42,43, using exactly their existing four posed cropped RGB inputs and K.
These are rendered observations of scanned assets. Stock generation uses view0;
registration uses all input masks; the geometry/photometric edit targets exclude
one camera-selected input view. Cached posed DA3 predictions can be reused only
after input/camera/view identities match; old NCC targets and old30K meshes or
latents are never used to supervise TRELLIS.2 edits. Recompute NCC from the new
base when needed. GT scans are evaluation-only, after input decisions are sealed.

Base comparisons: raw official output, the same output after input-only proper
similarity registration, and that registered base plus local latent editing.
Registration uses24 proper cube rotations followed by continuous similarity
refinement of the top3 mask fits. No GT ICP, reflection or separate alignment of
edited results. Retain low-registration-quality cases and mark them explicitly.

Technical smoke: exact native decoder parity with gradients enabled, finite
nonzero latent gradients, directional finite-difference checks on a fixed
discrete branch, and recovery of a small decoder-reachable local perturbation.
Synthetic targets/teacher latent are restricted to that clearly labeled smoke;
target latents never enter the optimization objective. Do not report smoke as
real reconstruction improvement. Optimizer, decoder and sampling caches are
separate. The initial fixed-topology recovery attempt reduced vertex MSE but
worsened native rendered depth, and is retained as a failed diagnostic. The
current optimizer redecodes the actual native mesh at EVERY update, using its
piecewise vertex gradients. Fixed discrete branches are derivative checks only.
Recovery version2 uses native rendered geometry loss and requires >=5% native
fixed-ray depth reduction, selecting by its generated-target training loss.

Natural repair: fixed local token set from input-supported depth discrepancies;
outside feature rows and SS coordinates frozen.40 attempted updates, cap0.25
native training std, Adam lr.02. Loss uses actual native mesh depth/silhouette and
outside protection, not a proxy latent loss. Final selection uses only input
target gain, observed-mask non-regression, exterior drift, heldout evidence and
component sanity. Bad inputs or no reliable evidence can deliver the base.

The official native representation supports open/non-manifold surfaces; do not
impose watertightness as a correctness condition. Report candidate and delivered
meshes, actual topology changes, elapsed time and peak VRAM. On3090, memory and
gradient viability are measured, not inferred from H100 inference timings.

This pilot does not train the independent pose_constrained_trellis2 Flow. It
does not change or restart the ongoing v1 round-ten experiment. Larger-cohort,
direct-mesh baselines,1024-resolution,SS edits and learned refinement follow
after native backend and input-only loop verification.
