# Trainable native T2 geometry repair, 2026-10-30

This snapshot adds a standalone geometry-repair training project. The first
denoising-flow Shape experiment failed its32-object overfit gate and is preserved.
Its old predicted-SS versus teacher-support gain field used mismatched input
support; read CORRECTED_INPUT_BASELINE.json and the package audit, not that field.

The residual bridge passed32-object training-data checks after600 updates.
Matched GT-support CD improved12.30%; full predicted-SS CD improved16.78% versus
the actual damaged sparse input. These are synthetic known-ROI overfit results,
not a Dev64 or real-generated-error claim. Both the original stock local prior
and a trained full multiview reconstruction control accompany formal experiments.
The independent inference smoke denied dataset/GT Python file opens and verified
OBJ/native NPZ readback.12 CPU contracts passed.

The original selection/caches remain intact. A new data-validity candidate pool
and formal split filter ambiguous geometry/render supervision before training,
using fixed ordering and no model scores. Long runs and all large artifacts are
kept under /data/zjr/pose_conditioned_geometry_repair on the original machine.
