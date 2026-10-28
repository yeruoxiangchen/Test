# Frozen shared-scene GT-support diagnostic

The completed SS/observation cross localized the dominant differences between
two image packages to SS support. This follow-up tests generated SLat geometry
on verified GT surface support, retaining all four SharedScene4 old objects.
It is a privileged diagnostic, not a learned method or a guaranteed strict
performance upper bound. It does not claim RGB-only operation, generalization,
training improvement or a LaRI contribution.

## Fixed comparison

Objects are chair i076, table i075, bench i080 and stool i065, with original
noise indices 0/5/6/7. Keep sampling seeds 42/43/44 and the four-UID frozen SLat
loading context. Fix crop2072_visible for all objects; source renders are 2072,
network inputs remain 518. Never select the best image package per object.

Reference: the 12 existing appearance_control predicted-support meshes using
this same image package. Oracle: 12 complete new SLat samples on GT support.
There are 24 logical cases, not 24 new or independent objects. Do not regenerate
SS, retrain weights, re-encode GT latent, prune predicted supports, hide the
failed bench geometry, or select a best seed.

GT support comes directly from the archived object ss.npz official_coords:
11136/9520/2652/8653 points on the canonical 64-cube surface. Verify it equals
the actual sample.target_coords and the coordinate set of slat.npz. It is not
the thresholded output of an SS decoder or a newly voxelized/filled solid.
Only coordinates are used to construct sparse_noise; no GT latent features
enter sampling. Strip supervision from image observations, then preserve the
frozen_slat INPUT_KEYS allowlist, full selected view set and camera conventions.

Keep frozen SLat30K EMA, original normalization, 25 steps, CFG5, interval [.5,1],
rescale_t3, positive25/negative19 calls, decoder, LargeInputNative and existing
memory-efficient mesh extraction. No precision, model, grid or resolution change.
Use master_noise(seed, original_index) indexed by canonical xyz. Common
coordinates share noise; support lengths/topologies may differ. Report empty
intersections explicitly instead of calling unequal noise arrays identical.

## Required gates and output

Freeze old identities, completed baseline, cross runtime, all reference/support
records, new code and this protocol. New statuses/caches/results are isolated.
All old outputs and frozen source files remain unchanged.

CPU audit: official coordinates, target encoding identity, coordinate sets,
grid identity, canonical transform, synchronized image/camera/condition views,
reference inputs, SS metrics, fixed GT partition counts, complete paired table,
and missing/duplicate rejection. No CUDA initialization during CPU preparation.

Short GPU gate: replay table/crop2072/seed42 with its predicted support exactly,
requiring equal canonical/world vertices, faces and metrics. Then complete
stool/seed42 on GT support; count it as the first of the 12 oracle meshes.
Check the 12 predicted/GT support noise pairs before generation. A replay,
identity, invalid output or cache binding failure stops later stages.

Archive sampled SLat before decoding, bound to support, observations, initial
noise, original index and exact model. Resume only after those contracts and
files validate; never reuse a prediction latent as the oracle result.

Use original native-frame target mesh and fixed shared-scene GT evaluation
partitions. No fitted rotation, translation, scale or ICP. Check canonical/world
export with the archived known transform, not predicted scene placement.
Report per seed and per object CD-L1, F@.02, SS occupancy, visible recall,
externally occluded recall and free-space violation; macro weight objects
equally. Seeds are within-object sampling repeats. Report paired oracle-minus-
prediction differences, ranges and failures; no automatic fit gate on positive
improvement, because true support does not mathematically guarantee better mesh.

Deliver 12 GT/predicted/oracle sheets using three original fixed views and six
four-object world GLBs (two support kinds x three seeds). GLB placement is known,
not inferred. Completion requires all 12 new meshes, 24 logical cases, replay
and noise gates, metrics and visualizations. Leave quality review explicit.

Use one selected GPU, default0 via HSSD_GT_SUPPORT_GPU. Read occupancy only; no
memory allocation probe or automatic GPU switching. The user authorized the
assistant to implement, check and launch; the user monitors long work. After
the short gate, launch the remaining queue and provide the absolute watch
command. Do not start SS training automatically from this diagnostic result.

## Interpretation

GT support is diagnostic access to complete surface location, not available
from a masked RGB crop. It changes point count, geometry and sparse attention
topology. If it improves mesh, prioritize learning better image-conditioned SS;
if errors remain, inspect the observation/SLat generator and existing codec
evidence separately. Historical pilot codec reconstructions are background
checks, not matched current image-generation results. Retain visible-region
accuracy and free-space constraints when designing later SS conditioning;
unknown/occluded space is not automatically empty. No new architecture claim
is established by this run.
