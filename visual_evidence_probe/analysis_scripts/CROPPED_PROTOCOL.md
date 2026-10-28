# Training-conformant crop control

2026-10-25 UTC. Added AFTER the full-frame probe exposed unstable geometry and
inspection found that its HSSD cache was full-frame, whereas the 30K training
contract uses foreground bbox crop with margin1.1. Preserve the first probe.

Run the FIRST FOUR objects from the same original ordered manifest, all eight
arms and both seeds. This is an explicitly post-hoc preprocessing control,
not a heldout test or an independently selected replication cohort. Do not
select further objects based on cropped results. Four objects yield64 meshes.

Use the exact shared-object preprocessing implementation, resolution518,
margin1.1, alpha threshold0.8, bilinear RGBA resize, black background and
K_feature=affine@K_original. Extrinsics/object coordinates are unchanged.
All original and synthetic images use the same frozen DINO in FP32 inference,
the original encode_image feature LayerNorm and FP16 cache storage.

For the two observed images, crop using their given visible object masks.
For each generated donor image, crop using ONLY that donor render's own mask.
Never crop a synthetic image with a hidden GT mask. Keep its matching cropped
intrinsics with its features. The GT-render arm remains explicitly privileged.
Mixed arms swap image AND corresponding intrinsics together. Camera centers
and the two donor-image marginals are identical across balanced averages.

Render/evaluate in the original eight full-frame cameras, preserving direct
comparability with original GT masks and native canonical geometry. No pose
fitting, scale correction or ICP. Reuse the frozen inference worker/equations,
SS25/SLat25, paired noise, fixed spconv kernels, memory-efficient decoder,
50000-point metrics and all failures. Run RGB/camera crop-affine projection
checks before inference. Freeze this protocol and all used sources.

Primary question: after restoring training preprocessing, does synthetic-input
harm remain, and does balanced donor mixing reliably help? No calibrated
geometric-confidence claim can be made. No automatic training follows.
