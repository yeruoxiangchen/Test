# Seventh experiment: posed dense depth and observed-evidence acceptance

2026-10-27. Frozen original30K SS/SLat endpoints, frozen mesh decoder, and
frozen DA3-LARGE-1.1 (official pinned revision). No model weights are trained.
All generated data, checkpoints, environments, caches and temporary files are
under /data/zjr/visual_evidence_probe. Prior experiment files remain unchanged.

Development: the same eight inspected Omni objects x two seeds.
Validation: sixteen deterministic, distinct object UIDs from the same Omni200
source; selection is frozen before inspecting their outputs or evaluation GT.
They are new to this local-repair development, not new to the original paper's
benchmark. Real pilot: the existing YCB-V drill's three annotated8-view groups
x two seeds, reported separately. BOP object poses, normalization from model
bounds, visible masks and annotated crop boxes are inherited assistance. Real
groups differ in camera coverage as well as visibility; not a causal occlusion
ablation or an unassisted-pose reconstruction benchmark.

Two input-evidence sources x two candidate policies. The same80-step editing
trajectory is shared between its two policies; only selection differs.
Input sparse uses the inherited NCC scheme. Dense augments those same trusted
NCC anchors with camera-conditioned DA3 depths on additional reliable rays;
it falls back exactly to sparse evidence when DA3 adds no local rays. This
development-stage adaptation was made after observing sparse DA3 coverage,
before inspecting any validation-object GT or editing results. Raw DA3-only
coverage and depth quality remain separately reported. Dense uses DA3 camera-conditioned
depth with the exact provided K and CV world-to-camera poses. Holdout images
are excluded from DA3 inference and sparse proposal construction. Dense depth
is mapped back using returned processed K and exact input-camera rays, not a
blind resize. Scale uses only the provided input camera trajectory; no GT mesh
ICP, shift, scale or per-view depth alignment.
Sparse proposals are freshly recomputed for the sealed split with camera-z
search offsets[-.06,.06]; this round does not add the old world-ray length clamp.
The new splits and fresh proposal pools mean round7 is not a paired replay of
round6 sparse targets. Comparisons within the four round7 arms remain paired.

Dense confidence, target foreground and forward/backward multi-view depth/
pixel consistency define a fixed evidence mask before optimization. Targets,
regions, editable rows, validation domains and all selection decisions are
input-only. GT is loaded only after decisions are sealed. No-prediction and
no-validation states remain in aggregate. Teacher quality is measured separately
with fixed coverage statistics and GT only as a posthoc diagnostic.

Legacy policy checks the own-training-selected checkpoint using the old input
photometric gate, original-mesh silhouette IoU>=.995, exterior drift and no
component-count increase. Revised policy checks every fixed checkpoint against
observed silhouette, fixed holdout photometric evidence, exterior drift and
area-weighted new small components. It does not require the corrected local
silhouette to match the original mesh. Rejected candidates restore the original
mesh and latent exactly. Revised policy cannot select by GT depth or metrics.
The revised gate requires at least16 fixed depth rays, at least8 locally visible
heldout pixels, >=10% training depth-residual reduction, heldout photometric
non-regression when measurable, and observed-mask non-regression in every view.
It may accept trusted depth improvement with unchanged heldout evidence; it
does not claim every accepted edit improves the heldout photometric cost.
Allowed additional small-component area is0.2% of surface area, with no new
large components or nonmanifold edges. Total area must remain within0.8--1.25
of baseline. These settings are fixed before reading validation-object metrics.

All arms share original latent/support, original source-only editable rows,
input-derived region, budget, feature bounds and loss weights. Input silhouettes
are supervised only where observation is informative; real-scene pixels with
predicted foreground occlusion remain unknown. Baseline geometry protects
unobserved parts and the exterior. New evaluation views never enter targets,
optimization or selection. Report both candidates and actual accepted outputs,
surface metrics, fixed-domain local/whole depth, topology, changes outside the
region, coverage, selected checkpoints, elapsed time and peak allocated memory.
Average seeds within objects before object bootstrap; the three real groups
share one physical object and are not three independent objects.

The heldout image is withheld from DA3, NCC construction, and this editing
loss. The original frozen30K mesh used all original inputs, so the initial
latent is not image-disjoint from that heldout view. Eight newly rendered
evaluation cameras remain excluded from every method decision.

Runtime note, 2026-10-28: two asynchronous edit smoke runs on GPU0 raised
device-side/illegal-memory CUDA errors. A fresh synchronous GPU7 run completed
both evidence trajectories on the affected development object. No algorithm
or data change was used to bypass the error. Resume incomplete input cases,
retain failure logs, and retry synchronously only for CUDA runtime failures;
never label failed optimization as a valid no-op. Record synchronization mode
separately from scientific parameters and interpret timing accordingly.
