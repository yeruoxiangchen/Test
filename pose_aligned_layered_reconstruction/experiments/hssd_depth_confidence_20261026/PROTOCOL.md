# HSSD depth × confidence diagnostic — 2026-10-26

## Question and fixed cohort

The visible-first-surface oracle improved F@.02 by 6.746 percentage points over predicted geometry, but replaced depth and confidence together. This final bounded mechanism diagnostic separates those changes before moving to held-out objects/scenes. It does not tune tau, filter objects/seeds or establish generalization.

Keep the same four old training objects (chair/table/bench/stool), original indices 0/5/6/7, crop2072_visible inputs and sampling seeds 42/43/44. Stool uses frames 0/11, all others 0/4/7/11. Do not drop bench failures. All cells use the same known cameras, visible target masks, canonical transforms and existing SS targets. No mesh-to-latent re-encoding or VGGT inference is needed.

| Cell | Depth | Confidence | Action |
|---|---|---|---|
| geometry_observed | Frozen RGB prediction | Frozen per-view empirical rank | Reuse completed SharedSS |
| geometry_gt_depth_pred_confidence | GT visible first hit | Frozen per-view empirical rank | New 200-step training |
| geometry_pred_depth_unit_confidence | Frozen RGB prediction | One on valid observations | New 200-step training |
| geometry_oracle_visible | GT visible first hit | One on valid observations | Reuse completed oracle |

The primary factorial has 4 cells × 4 objects × 3 sampling seeds = 48 results. Completed appearance_control adds 12 context references. Total: 60 logical results, 24 new SS/mesh and 36 archived references. Appearance is not a fifth factorial cell.

## Evidence and leakage boundary

Use the already frozen group4/group2 predicted `depth_z_world` and `rank_conf` arrays; retain their original scale determined only by known/predicted camera baselines. The confidence rank is not a calibrated probability. Unit confidence removes both rank-based rejection and attenuation; it is not a claim of certainty or a new confidence estimator.

GT depth comes exclusively from the previous oracle's stripped first-surface caches, whose exact fields are depth_z_world and valid. These are full-scene **visible** first hits plus the same target-visible masks: they include observed occluder surfaces and front free space, never hidden target geometry. Do not read instance IDs, amodal/target-only depth, complete GT surfaces or target support as conditioning. GT latent remains training supervision only.

Validity belongs to the chosen depth source. The predicted-depth/unit-confidence arm must not read GT valid masks or GT depth receipts. Invalid selected-depth observations, including any nonzero invalid contribution to bilinear interpolation, remain unknown with confidence zero. Depth/rank are bilinear; visible masks are nearest resized 2072→518. Within the same fixed surface tolerance band, retain the original soft feature semantics; beyond observed first surfaces, hidden target geometry is unknown. No hard carving or new loss is introduced.

Reuse exactly the original projector, fine_evidence and pool_xyz: 64³ fine grid, 16³ tokens, twelve channels in the original order, tolerance 2/64, rank threshold .3, xyz ordering and weighting. Before building new conditions, recompute both archived diagonal cells for all four objects and require **torch.equal** on features, weights and valid (8 cases). If any differs, stop before GPU; no tolerance relaxation or silent reference replacement.

## Matched model, training and sampling

Both new cells use the identical SharedEvidenceSS geometry adapter with 3,164,160 trainable parameters and the complete frozen old appearance parent (including its residual and no_layer base). Start each adapter independently at seed20261026 with the same zero projections as all three archived controls. Neither smoke weights nor another factorial arm initialize a formal run.

Each new arm trains 200 steps, batch1, AdamW lr1e-4/warmup20/betas(.9,.95)/weight decay.01, BF16, clip1, raw weights, no unconditional drop, original velocity MSE only. Require identical input/target hashes and the whole 200-step object/index/time/noise/LR sequence; each object receives50 exposures. No extra parameter or optimization-budget advantage.

SLat remains frozen 30K EMA. Preserve SS/SLat seed formulas, 25 steps, CFG5 on [.5,1], rescale_t3, 25 positive/19 negative calls, deterministic native operators and identity grid. No per-result scale/translation/rotation fitting or ICP. Export canonical and world meshes using the fixed canonical_from_world.

CPU preflight validates actual cached input/target equality and all36 references. Actual GPU gates test each new arm's zero-injection equality/gradient isolation and independent two-step training against all archived schedules; one complete original table2072/seed42 SS+mesh replay checks the common unchanged generation path. Both arms' zero-output checks establish equality through that common path. Only a complete bound smoke receipt admits formal training.

## Analysis fixed before results

Report per-object/seed metrics and equal-object macro means; primary conditional effects are:

- Depth effect at predicted confidence: GT/pred − pred/pred.
- Depth effect at unit confidence: GT/unit − pred/unit.
- Confidence-policy effect at predicted depth: pred/unit − pred/pred.
- Confidence-policy effect at GT depth: GT/unit − GT/pred.
- Interaction: (GT/unit − pred/unit) − (GT/pred − pred/pred).

For CD and free-space violations lower is better; for F/IoU/recall higher is better. The interaction is a descriptive paired difference of differences, not statistical evidence from independent training repetitions. Keep all seeds and negative outcomes. Report F@.01/.02, CD, normal consistency, SS precision/recall/IoU, visible/occluded recall and free-space violations; reuse the original GT partitions.

Produce12 sheets containing GT and all5 methods under the same three cameras, plus15 known-placement world GLBs. Inspect topology as well as metrics, especially bench and table failures. Do not label a recall increase caused by large false surfaces as successful completion.

This is the last planned four-object mechanism diagnostic. No follow-on threshold/budget change is automatically launched. After review, freeze one RGB-derived condition and move to a scene/object-disjoint evaluation plan with matched appearance as main baseline. Oracle cells stay diagnostic. This study alone does not justify restarting original LaRI long training.

## Storage and execution

All new outputs, checkpoints, temporary files and runtime/model caches stay under `/data/zjr/pose_aligned_layered_reconstruction/`. Reuse the prior verified data mapping and offline model cache; copy only the additional predicted depth/rank and diagonal evidence files. Preserve original logical hashes; do not scan raw scenes, reload the5GB VGGT model or alter other active jobs.

Single-GPU stages select an idle card first; if none is idle, allow a card with at least18432MiB free, choosing the largest headroom. Read-only nvidia-smi checks only, no allocation probes. Re-select between completed stages rather than waiting forever on an occupied fixed card; never switch during a running stage or kill another job. Insufficient headroom waits with heartbeat. Failures stop the queue and retain completed bound artifacts.

Queue: storage → CPU input audit → diagonal replay and crossed evidence → GPU smoke → two independent200-step runs →24 SS →24 mesh → summary. The user monitors long work; the agent starts the prepared queue and reports the absolute watch command.
