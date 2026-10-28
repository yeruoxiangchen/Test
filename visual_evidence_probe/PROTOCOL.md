# Correlated synthetic-view evidence: initial diagnostic

Created 2026-10-25 UTC, before this probe's inference results.

This is a bounded hypothesis check, not a training run or a paper benchmark.
Use the first eight records of the existing HSSD clean16 manifest, in its saved
order. These are eight previously inspected objects in one rendered scene;
they cannot establish held-out real-world generalization. No LaRI weights are
used. Freeze the original SS30K EMA, SLat30K EMA and Stock mesh decoder.

For each object choose view 0 and the most angularly separated remaining camera,
using camera centers only. The other six cameras are fixed for every arm. Make
two donor meshes from these same two source images, using seeds 42 and 43.
Render each donor and the original GT mesh under the SAME neutral-gray renderer,
lighting, camera intrinsics, extrinsics, image size and black background. The
GT render arm is explicitly privileged and diagnoses rendering-domain effects;
it is not a deployable method. Do not align, rescale, rotate or ICP any result.

Arms:

1. real2: the two original, masked, Cycles-rendered observations.
2. repeat8: repeat the same two original observations four times each.
3. real8: all eight original observations, with source views first.
4. gt_render6: real2 plus six gray GT renders (oracle domain control).
5. donor_a6: real2 plus six renders of donor seed42.
6. donor_b6: real2 plus six renders of donor seed43.
7. mixed_ab6: alternate donors A/B at the six fixed cameras.
8. mixed_ba6: complementary donor allocation at the same cameras.

The mixed arms and single-donor arms have identical camera coverage, counts and
renderer. Across their paired averages they use the same donor-image marginals;
this controls the quality of any one donor better than a single A-vs-mixed
comparison. Different seeds are DIFFERENT HYPOTHESES, not proven independent
evidence. All donor images depend on the same two original observations.

Use final sampling seeds 42 and 43 with identical per-object master noise across
arms. SS25 + SLat25, original CFG and normalization. Save OBJ meshes, SS support,
input images, source identities and checkpoint hashes. Use the existing reviewed
fixed native-spconv kernel policy for ALL arms and the existing memory-efficient
mesh reduction. Test identical-input replay before interpreting small effects.

Score against original canonical HSSD geometry: surface CD/F@.02, SS surface
occupancy IoU, fixed-camera silhouette IoU, and donor-directed surface/support
agreement. Seeds are paired repetitions; objects are the aggregation unit.
Surface metrics use 50,000 points and fixed evaluation seed 20261025. Donor
agreement uses 20,000 points. Verify independently rendered GT masks against
saved masks before sampling; require every view's IoU >= .90 (no fitted pose).
Report every failure, do not filter bad outputs, and do not silently retry with
different seeds or easier objects. An interrupted exact job may resume.

The parent has NO calibrated geometric confidence head. SS decoder-logit
magnitude or sigmoid values must NOT be called confidence. This probe can test
geometry degradation and donor agreement, not the full 'more confident yet more
wrong' claim. Multi-seed output agreement is a separate descriptive statistic.

Interpretation:

- Worse donor arms than real2 alone only show synthetic-input harm, not its cause.
- Compare donor arms with gt_render6 to diagnose rendering/input-domain mismatch.
- Compare the balanced single-donor average against both mixed arms to probe
  source correlation while matching donor marginals. Eight objects/two donors
  cannot establish a universal causal law or statistical independence.
- If repeat8 differs, inspect context-token subsampling and numerics before
  claiming repeated evidence changes geometric belief.
- No automatic expansion or new module training follows this diagnostic.
