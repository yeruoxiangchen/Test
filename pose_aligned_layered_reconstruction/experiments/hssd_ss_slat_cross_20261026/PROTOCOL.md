# Frozen SS support × SLat observation crossover

Purpose: distinguish the support-mediated and SLat-observation-mediated changes
seen in completed InputAdapt4. This is a post-hoc diagnostic, not new training.
The selected old training objects are table i075 and stool i065 in the same HSSD
scene. Retain GT-assisted cameras, visible masks and canonical frames; do not
claim held-out generalization, RGB-only reconstruction or a LaRI contribution.

## Frozen factorial design

Use the existing appearance_control SS supports, seeds42/43/44, and input levels
crop518_visible / crop2072_visible. Cross SS support source with the complete
SLat observation package (masked visual features, all selected views and camera
metadata). There are 2 objects × 3 seeds × 4 cells = 24 logical positions:
12 old diagonal mesh references and 12 new off-diagonal complete SLat samples.
Do not resample SS, rebuild GT latent, train weights, drop objects or select a
best seed. Do not count the diagonal references as new independent experiments.

All four archived UID contexts remain passed to load_frozen_slat30k, preserving
the exact historical slat_binding (evaluation_uid_count=4). Only table/stool are
sampled here. Original object indices are 5 and 7, never subset indices0/1.
Retain the exact SLat30K EMA, normalization, 25 steps, CFG5, interval[.5,1],
rescale_t3, positive25/negative19 calls and unchanged decoder.

Noise is the original master_noise(seed, original_index), indexed by canonical
xyz in the64³ grid. A common coordinate receives exactly the same noise. Distinct
supports need not have equal size or overlap (stool seed43 has no overlap).
Do not call this equal-length or identical noise vectors across support sets.

Use the already validated LargeInputNative policy (ordinary tensors delegate to
FixedNative), the existing memory-efficient mesh extraction and native frame.
Keep every support point. No precision, channel, resolution, fitted rotation,
translation, scale or ICP change. Save the exact sampled latent before decoding,
bound to both the support and observation sources, to resume without resampling.

## Integrity and short checks

All old experiments and their completed output remain immutable. New code,
outputs, statuses, cache receipts and queue.lock live only in this experiment.
Freeze the old plan/evaluation identity, final completion/summary, native runtime
amendment and regression gates, all12 SS+diagonal records, and all four original
SharedScene evaluation_gt depth/owner arrays. Bind new source and protocol files.

CPU preparation checks actual low/high caches: same object/GT/target coordinates,
K, extrinsics, image size, grid and canonical transform; differing masked DINO;
synchronized views (table0/4/7/11, stool0/11). Supports must be unique sorted
int32 canonical coordinates in64³. Recompute SS occupancy against the same GT.
Track support_source and condition_source independently. Crossing intentionally
means their image-input bindings differ.

Before the long queue, perform a GPU coordinate-noise check and two full samples:

1. table seed42, support518/condition518: replay the archived diagonal. Require
   identical canonical and world mesh vertices/faces after export and reload,
   full surface/structure/partitioned metrics, SLat binding and sampler calls.
2. stool seed42, support518/condition2072: a genuine cross with two selected views.
   This is the first of the12 new meshes and is reused by the long queue.

Any identity, replay, empty/nonfinite output or sampling error stops the queue;
the difficult case remains missing/failed. Existing failed gates cannot silently
be treated as passed. Validate source/condition/cache contracts on resume.

## Evaluation and deliverables

Use the original surface/structure/partitioned metrics and original shared plan,
with no fitted transform. Keep GT20k surface sampling/partitions fixed:
table visible5247 / externally_occluded9477 / other5276;
stool visible5691 / externally_occluded14309 / other0.
SS IoU must be identical across the two conditions at any fixed support.
Report CD↓, F@.02↑, visible/occluded recall↑, free-space violation↓ and support
statistics. Free-space violation is not collision volume.

For each object and seed, report both condition effects at fixed support, both
support effects at fixed condition and interaction HH−HL−LH+LL (first axis SS).
Keep raw high-minus-low signs; negative CD and positive F indicate improvement.
Average three paired sampling seeds per object, then the two object means.
Do not infer independent-scene significance from these24 slots.

Reuse the12 diagonal renders and two GT renders; render only12 crosses with the
same three native cameras/material/scale. Deliver6 GT+four-cell comparison sheets
and12 two-object world GLBs (four cells×three seeds). Layout is archived known
placement, not model-estimated scene layout. Final completion requires all12 new
meshes, all24 logical cells, the replay gate, metrics and visualization artifacts.

SS support changes occupancy count, geometry and sparse attention topology.
SLat observations change both global and projected DINO; this experiment does
not isolate those subbranches. Low/high renders also differ in antialiasing and
mask rasterization, so the result is not a pure resolution effect. Preserve any
interaction rather than forcing a single-stage explanation. GT-support oracle
sampling, additional scenes and model training are outside this run.

Use one explicitly selected GPU, default4 via HSSD_CROSS_GPU; query its occupancy
without allocating a memory probe. Codex completes short checks. The user starts
and monitors the remaining long queue. Never announce12/12 completion before the
artifacts exist. Continue time-stamped project records without editing old frozen
reports or new frozen sources while a queue is active.
