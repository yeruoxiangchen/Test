# V3 native geometry-guided Flow

Current run: `/data/zjr/trellis2_geometry_repair/outputs/guided_flow_v3_fixed_20261029`.
The initial preflight is preserved separately, with its protocol/source archive.
V2 and the native pilot are frozen dependencies, not overwritten results.

The last three native Euler calls use differentiable decoder surface fields,
child-route logits, input depth/silhouettes and geometry preservation to update
the predicted clean latent. The Euler velocity correction respects upstream
sigma/time conventions. Model weights stay frozen. The Flow model is offloaded
during decoder backward; actual transfer time and peak memory are measured.

The corrected objective keeps all positive queries and protection anchors in
the denominator and freezes missing-support penalties for each backtracking
step. Disappearing children/cells cannot create a lower positive-target cost.
Fine fields use oriented edge crossings, not a fabricated voxel occupancy.

Controls share the same frozen SS, ROI, image context and proposal noises.
Compare two unguided proposals, an input-cost-budgeted unguided bank and two
input-guided proposals. After sealing actual-input choices, run paired input
and GT-target diagnostics on the same GT-intersection domain. Both paired
arms are diagnostic-only; the actual method does not use the GT-domain mask.

```bash
bash trellis2_geometry_repair/run_guided_flow.sh tests
bash trellis2_geometry_repair/run_guided_flow.sh launch t2-guided-v3fix-gpu4-r2-20261029 --gpu 4
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run_guided_flow.sh status'
```

Use `--aux --indices 10,11` after a launch's GPU argument for a spare-GPU worker.
It waits for the first primary case to finish and uses the same per-case locks.
The primary queue owns global artifact checks and audit closeout. Foreground
commands also expose `generation`, `selection`, `evaluate`, `report` and `smoke`.

`T2_GUIDED_PROFILE=initial ... status` inspects the stopped initial preflight.
Reproduce its numerical behavior from its source archive rather than running
the corrected code into its frozen directory. See PROTOCOL.md for all gates,
timing scope, inherited paths and the known four-object development limitation.

## Completed result

All8 cases completed at 2026-10-29T08:12:02+00:00; 112 latent states and36
composites passed artifact validation. Input accepted1/8: delivered CD gain
0.1177% vs base and0.0167% vs budget-matched prior, with mixed F-scores and
previous-good-region regression. Technical validation does not establish
reliable repair. Read data-side FINAL_ANALYSIS.md and PAIRED_GEOMETRY_AUDIT.md
and audit §6.3. Initial/failing preflights remain separate and preserved.
