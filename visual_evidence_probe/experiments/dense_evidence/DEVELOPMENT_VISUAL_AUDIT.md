# Development visual audit, 2026-10-28 UTC

Reviewed the saved same-frame render `figures/meshes/005_42.jpg` after both
seeds of `omni_brussels_sprout_002` had been sealed and evaluated. The four-arm
parameters were kept fixed; no validation-object GT informed these settings.

The NCC+DA3 candidate/accepted mesh retains the basic round object shape, but
introduces visible short ridges/creases near its upper surface. The new-view
local camera-z metric improves on this object, while its global Chamfer and
F-score worsen. These outcomes should be reported together. A positive local
depth result and an input-only gate pass do not establish a better complete
mesh or better surface smoothness. This note covers one development object;
it is not a visual review of the complete cohort.

The figure uses identical new evaluation cameras, shared cropping and lighting
for GT, frozen30K, both training-selected candidates, and both revised-policy
accepted outputs. The GT scan is an evaluation reference, never an editor
target in this experiment.
