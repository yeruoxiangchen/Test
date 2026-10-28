# Development pilot visual audit, 2026-10-28

Reviewed `005_42` in the saved shared-camera grid. Both constrained arms retain
the basic object geometry and their global CD changes reverse the degradation
of the round-seven reference on this single seed. However, short upper-surface
ridges remain visible in all edited rows. The new terms do not establish a
surface-smoothness solution. All geometry parameters remain frozen for the
ongoing development experiment; the visual observation is not a validation
result or a reason to change hyperparameters on later evaluated objects.

The real-drill round-seven render also shows substantial baseline global
registration discrepancy. This is consistent with its input-domain silhouette
IoU around0.12--0.29, while the annotated GT projection matches full masks at
about0.982. The coordinate/depth audit and local-editing experiment are kept
separate; no GT registration is used to improve method scores.
