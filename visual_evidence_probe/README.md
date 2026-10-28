# Correlated synthetic observations: bounded hypothesis probe

Inference-only diagnostic of whether multiple images of the same generated
hypothesis amplify geometric errors. No model training or changes to the paper
endpoint. See [PROTOCOL.md](PROTOCOL.md) for pre-result choices and limitations.

```bash
bash visual_evidence_probe/run.sh smoke
bash visual_evidence_probe/run.sh run --gpus 0,1,4,5
bash visual_evidence_probe/run.sh status
```

Outputs: `outputs/correlated_views_20261025/`; per-object input galleries, OBJ
meshes, GT bindings and numerical records. Final aggregation writes `REPORT.md`
and `summary.json`. The first eight saved HSSD objects are exploratory examples,
not held-out validation. GT-rendered inputs are an explicit oracle control.

The probe reads maintained frozen runtime helpers from existing packages; it
does not copy or retrain their models. Kernel selection is fixed using the
existing process-local inference policy. Check identical-input replay before
running the queue. The launcher uses only explicitly listed physical GPU IDs.

The first full-frame run exposed a preprocessing mismatch with the parent30K
foreground-crop contract. Its outputs remain archived as exploratory diagnostics.
The explicitly post-hoc, training-conformant control uses the first four objects,
all eight arms and both seeds (64 meshes):

```bash
bash visual_evidence_probe/run_cropped.sh run --gpus 0,1,4,5
bash visual_evidence_probe/run_cropped.sh status
```

See [the crop protocol](analysis_scripts/CROPPED_PROTOCOL.md) and
[audit](审计.md). Cropped results are under
`outputs/correlated_views_cropped_control_20261025/`. Each synthetic view uses its
own foreground mask and matching transformed intrinsics; no hidden GT crop is used.

The second diagnostic adds input-only silhouette selection, fixed rejection
thresholds and proper global similarity fitting. It keeps the old sources and
outputs intact; all four inspected objects and both seeds are retained.

```bash
bash visual_evidence_probe/run_consistency.sh prepare
EVIDENCE_GPU=4 bash visual_evidence_probe/run_consistency.sh run
```

Outputs: `outputs/input_consistency_baselines_20261025/`. See its `REPORT.md`,
`summary.json` and [protocol](experiments/input_consistency/PROTOCOL.md).
An additional control applies the same final fitting to all pseudo-view outputs,
so comparisons with fitted two-view outputs control global registration:

```bash
CUDA_VISIBLE_DEVICES=4 /home/zjr/anaconda3/envs/reconviagen/bin/python -m visual_evidence_probe.analysis_scripts.output_alignment_control
```

Its outputs are under `final_output_alignment_control/` inside the second run.
Both protocols are exploratory follow-ups on the same rendered scene. Fitting
and source selection use only the two observed masks and cameras; GT-only
oracles are explicitly diagnostic.

Read the completed [second experiment conclusions](第二次验证结论.md) and
[timestamped audit](审计.md) before proposing model training.
