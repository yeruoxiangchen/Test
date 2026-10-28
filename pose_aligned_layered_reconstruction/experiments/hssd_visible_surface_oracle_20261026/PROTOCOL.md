# HSSD shared-scene visible-first-surface oracle — 2026-10-26

## Question and scope

The preceding SharedSS study improved mean F@0.02 but did not consistently improve occluded-surface recall; its bench has nearly no reliable predicted target-surface evidence. This experiment asks whether accurate **observed first-surface depth and confidence**, under the identical evidence operator and residual architecture, improve the matched result. It is a four-object training-set feasibility diagnostic in one scene, not a generalization result or a LaRI contribution claim. Three sampling seeds are repeats within each object, not independent training runs.

Only `geometry_oracle_visible` is newly trained. Completed `appearance_control` and `geometry_observed` results from `hssd_shared_ss_20261026` are frozen references. The primary table has four objects × three seeds × three methods = 36 cases: 12 new SS/mesh and 24 historical references. Do not drop failures or rerun favorable seeds selectively.

## Conditioning boundary

Use the same crop2072_visible cached DINO/RGB conditions, selected frames, known cameras, visible masks, canonical coordinates and latent targets. Chair/table/bench use frames 0,4,7,11; stool uses only 0,11. GT assistance in cameras/masks/canonical frames is unchanged.

Extract **only `depth_z_world`** from frozen full-scene first-hit render archives into separate caches with exact fields `depth_z_world` and `valid`. Evidence generation and training consume these stripped caches. They do not read instance IDs, amodal masks, target-only depth, complete target surfaces, mesh occupancy or target latent as conditioning. Full target data remains legitimate training supervision and evaluation ground truth.

Valid positive depth receives confidence one. Invalid/nonfinite/no-hit depth is unknown; any invalid contribution to bilinear interpolation sets the observation confidence to zero. Depth remains bilinear, visible masks nearest, using the old projection implementation. The first surface and front free space contribute soft evidence; behind it remains unknown. No hard carving is introduced. This jointly replaces predicted depth **and confidence**; it is neither a pure depth ablation nor a guaranteed mathematical upper bound.

Reuse the exact prior 64³ grid, 2/64 canonical tolerance, 16³ pooled tokens, twelve-channel ordering and confidence threshold 0.3. The internal geometry adapter class is unchanged. No new loss, hidden-geometry label, or latent encoding pipeline is introduced.

## Matched training and sampling

The parent is the full old 200-step occluded appearance-control model, including its no-layer base and trained residual. The new adapter starts from the same seed 20261026 and exact zero-projection initialization as both SharedSS controls: 3,164,160 trainable parameters. Train 200 steps, batch one, AdamW lr 1e-4, warmup 20, betas (0.9,0.95), weight decay .01, BF16, gradient clipping one, velocity MSE only, raw weights, unconditional probability zero. Preserve object order, noise, diffusion times and all original indices; each object receives 50 exposures.

SLat remains the frozen 30K EMA model. Keep the previous SS and SLat seed formulas, seeds 42/43/44, 25 sampler steps, CFG 5 on [.5,1], rescale_t 3, 25 positive and 19 negative calls, deterministic operators and native identity grid. Do not use per-result rotation, translation, scale fitting or ICP. Export canonical and world meshes with the one frozen canonical_from_world mapping.

Before formal training: CPU source/target validation; evidence invariants including invalid observations; actual CUDA zero-adapter parent-output equality and gradient isolation; separate two-step training with exact historical schedule/target/noise comparison; full table2072 seed42 SS **and mesh** parent replay. Formal training starts independently from zero, never from the smoke checkpoint. A failed check stops the queue.

## Evaluation and decision

Report per-seed pairs and equal-object macro means for CD, F@.01/.02, normal consistency, SS IoU/precision/recall, visible/occluded surface recall and front-free-space violations. Keep original GT partition denominators and report missing strata as null. Produce all 12 comparison sheets (GT/appearance/predicted/oracle), plus world-coordinate meshes. Inspect shape quality and consistency across objects/seeds; training loss alone is insufficient.

If oracle improves consistently while predicted evidence does not, prioritize observation quality, confidence calibration and coverage. If oracle also fails to improve under matched training, revisit the geometry operator/injection and investigate whether the training budget suffices; this is not proof that all geometry or LaRI directions are impossible. Before paper-scale claims, require held-out objects/scenes and RGB-estimated camera/mask robustness. This diagnostic itself is privileged and cannot be reported as the deployable RGB method.

## Storage and execution

Source and small documentation stay in Tracker. All new outputs, checkpoints, staging, temporary data and runtime/model caches live under `/data/zjr/pose_aligned_layered_reconstruction/`. Old logical SHA bindings remain unchanged via an explicit storage manifest. Copy only required old payloads once with low I/O priority, verify while copying and verify the destination; retain original home files because other experiments may use them. Do not rehash or scan raw assets, the 5GB VGGT model, or entire home caches. Four required TRELLIS stock models are copied into the new offline HF cache; existing 30K checkpoints remain on /data.

The queue waits without allocations for an allowed GPU with no compute PID and memory below 256 MiB. It chooses one physical card and retains it across stages, waiting again if another task occupies it between stages. The queue runs storage → CPU preparation → depth extraction → evidence → smoke → train → SS → mesh → summary, with process lock, heartbeats and resumable bound artifacts. The user monitors long work through the experiment's `run.sh watch`.
