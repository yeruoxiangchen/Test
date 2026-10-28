# Sixth experiment

Frozen30K local latent editing with four geometry-evidence treatments. The full
protocol and distinctions between privileged oracle candidates and input-only
outputs are in [PROTOCOL.md](PROTOCOL.md).

Physical outputs: `/data/zjr/visual_evidence_probe/outputs/geometry_evidence_20261027/`.

```bash
# The two worker partitions used in this experiment. Completed cases are reused.
bash visual_evidence_probe/run_geometry_evidence.sh run --gpu 0 --indices 4,6,8,10
bash visual_evidence_probe/run_geometry_evidence.sh run --gpu 1 --indices 5,7,9,11

bash visual_evidence_probe/run_geometry_evidence.sh tests
bash visual_evidence_probe/run_geometry_evidence.sh report --final
bash visual_evidence_probe/run_geometry_evidence.sh figures --gpu 0 --overwrite
bash visual_evidence_probe/run_geometry_evidence.sh closeout
```

Each `objects/INDEX/SEED/ARM/` contains:

- `proposal.obj`: fixed80-step endpoint.
- `selected.obj`: best checkpoint by that arm's own training evidence.
- `accepted.obj`: selection passing the input-only deployment gate, otherwise
  exactly the original output. Oracle candidates remain privileged even when
  their gate uses only input evidence.
- Corresponding `*_latent.pt`, optimizer curves, input-gate receipt and metrics.

`figures/meshes/`, `figures/errors/` and `figures/inputs/` give16 matched views of
each kind. Mesh/error panels share cameras and a common display crop across all
rows; they do not perform per-mesh alignment or resizing. New cameras are used
only in final evaluation. `completion.json` verifies seals, unchanged support,
frozen feature rows, feature bounds, exact rejected-output fallback and all
required artifacts. `closeout` appends to the audit and preserves its old bytes.
