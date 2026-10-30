# Residual bridge amendment — 2026-10-30 UTC

Round1 is preserved under `outputs/round1_20261029`; its Shape gate failed.
The corrected physical-input baseline audit is separate. The new experiment
is `outputs/bridge_v2_20261030`; it is not a renamed Round1 success.

Hold SS600, the 32 overfit objects, cameras, ROI interventions and native Shape
targets fixed. Change only the Shape repair trajectory and its velocity target:

- The interpolated state moves from damaged at t=1 to clean at t=0. Network
  inputs add 0.1*t*fixed noise, while integration maintains the unaugmented state.
- Predict adapted native flow minus the same frozen official flow. Supervise
  this residual against damaged-minus-clean features. Native timestep is 650*t.
  Half the training samples are at t=1; the remainder use uniform t.
- At zero initialization the operation returns the damaged input exactly,
  including under different augmentation seeds. The latent-loss reference is
  this unchanged input, not the old stock denoising velocity target.
- Twelve Euler steps; exact copying of input features outside the edit region.
  Native FDG decoding may still change boundary geometry.

Use both the GT-support diagnostic and SS-predicted rollout. For missing parts,
the actual damaged reference omits absent sparse cells. The zero-feature tensor
with GT coordinates is a separate teacher-support diagnostic, never the physical
input baseline. Keep official stock local sampling as an independent mesh control.

Overfit gates remain: edit loss <95% of its stated reference; teacher-support
mesh CD lower than its matched-support damaged control; exterior features exactly
unchanged; predicted-SS mesh CD lower than the actual damaged sparse input.
Evaluate all 32 after 600 updates, then resume to 2000 if needed. No passing
claim from the one-object five-update runtime smoke.

If these gates pass, continue the already-authorized formal experiment: native
Train2000 / UID-disjoint Dev64, fresh official initialization, four training jobs
of 4000 updates / accumulation4. Repair SS and full-reconstruction SS/Shape use
standard flow matching; repair Shape uses the residual bridge. Model/data
selection is not based on Dev64 scores. Train view subsets and interventions are
deterministic by data position, independent of objective-specific RNG draws.

Formal evaluation uses all four corruption kinds, including identity, seeds42/43.
Report actual predicted-SS reconstruction, GT-support controls, native codec floor,
per-kind results and timing. Improvements on synthetic corruptions with known ROI
are not evidence of automatic localization or correction of real generative failures.
Include `stock_predicted_ss`: unmodified SS and Shape flows with their own support
and the same reference/ROI, in addition to the stock Shape / GT-support diagnostic.
The formal full multiview reconstruction control remains a separately trained arm.

Pure inference consumes only provided posed features, reference latents, ROI and
adapter checkpoints. Its runtime smoke forbids dataset/GT Python file opens and
verifies the exported OBJ and native NPZ roundtrip. It does not provide a score.

Formal data amendment: keep the original selection and all overfit artifacts.
Validate the complete available native candidate pool (2112 source-train +64
source-val). Retain the preregistered voxel-tolerant projection-support threshold
of .8; record/exclude ambiguous supervision instead of aborting the full builder.
Freeze a new formal_plan.json before training: first2000 valid source-train in
the original hash order; prefer valid original validation UIDs, fill missing
Dev64 slots from unused valid reserves. Exclude all formal-training and overfit32
UIDs from Dev64. No model output/metric enters this selection. Legacy caches pass
the same visibility criterion, with unchanged target tensors and pose hashes.
