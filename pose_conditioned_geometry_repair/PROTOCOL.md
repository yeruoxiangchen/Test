# Round 1: trainable local geometry repair (2026-10-29 UTC)

This is a synthetic intervention experiment establishing whether a trainable
TRELLIS.2 repair operator can use calibrated multiview evidence. A known synthetic
ROI is supplied. These results alone do not establish automatic error localization
or improvements on unconstrained generated errors.

## Data and initialization

- 2,000 unique Objaverse training objects, 64 UID-disjoint development objects.
  Preserve the existing native-label train/validation membership. Select training
  UIDs by a fixed hash, never reconstruction scores. The first 32 training UIDs
  form the overfit check. Frozen plan: `/data/zjr/pose_conditioned_geometry_repair/datasets/native_objaverse_v1/plan.json`.
- Native Shape512 labels: `microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16`,
  32 channels, 32³ sparse coordinates, at most 8,192 tokens. Existing frozen encoder
  posterior samples are reused. Three over-limit objects are excluded by schema.
- SS targets are freshly encoded from GT surface occupancy64 using the native SS
  VAE. The old rendering cache provides occupancy coordinates and canonical
  metadata; its `z` is never loaded. This differs from the official toolkit's
  Shape1024-support-derived SS labels. Measure the SS32/Shape32 support mismatch
  explicitly; never hide it by treating the two supports as identical.
- Eight original posed images per object; DINOv3 ViT-L/16 at 512 pixels, all 1,029
  tokens including five prefix tokens. RGB crop and K use the same affine.
  Cameras are explicit OpenCV w2c (+z forward, +y down) in the native source-mesh
  canonical frame. No implicit axis guessing or per-method GT alignment.
- Initialize both SS and Shape flows from official TRELLIS.2 weights. Neither
  old 30K flow checkpoints nor distilled targets are loaded. Freeze the backbone;
  train multiview aggregators, 30 zero-initialized block projections, attention
  LoRA rank 8 / alpha 16, and a damaged-state condition MLP.

## Interventions and model

SS input states include a locally deleted part, locally displaced surface support,
a spurious surface shell, and identity. Each corrupted occupancy is encoded using
the same native SS VAE. All changes remain inside the recorded ROI64. Shape input
features are deleted, replaced/displaced, perturbed, or unchanged in that same
ROI. Shape features outside it are exactly preserved. These are latent/support
corruptions; they are not claimed to be a realistic distribution of generated
mesh failures or physically rendered occluders.

The reference-state MLP sees corrupted features, editable-region mask, feature
presence and canonical xyz. It is combined with posed multiview features and
injected into the frozen native flow through per-block adapters. The original
single-reference-view DINO context stays in cross attention. Dropping images for
CFG retains the reference geometry. No clean target or GT depth is a condition.

Train rectified-flow velocity matching. Repair loss measures edit and exterior
regions separately (SS exterior weight .25, Shape .1). The full-reconstruction
control trains the same pose-conditioned model with zeroed reference-state input
and uniform full-state loss, then samples from full noise. It cannot inspect
corrupted features or ROI. Repair starts at t=.65; Euler 12 steps; CFG=1 in this
first experiment. Shape reference features are clamped outside the ROI along the
same noise path and copied exactly at the endpoint. SS decoded occupancy is
composed with reference occupancy outside the ROI. Native FDG decoding can still
alter boundary mesh vertices: exterior latent equality is not a guarantee of
pointwise mesh preservation.

Shape supervision uses native clean support, an explicitly teacher-forced
training condition. Rollout composes predicted SS support inside the ROI with
the **input** Shape support outside it; feature lookup copies only exact input
coordinates, with missing features flagged. The full reconstruction control
uses its entire predicted SS support. Report teacher-support and predicted-SS
mesh metrics separately. Never fill missing predicted features from GT.

## Execution and gates

1. CPU contract checks; native GPU training, gradient, zero-init and checkpoint
   smoke; resume from update 3 to update 5.
2. Overfit 32 fixed missing-part interventions, all eight views. SS 600 updates,
   Shape 1,000, batch size one. If a stage fails its gate, resume to 2,000 updates.
   Fixed evaluation noises are not training noises.
3. SS gate: edit-region flow loss <95% of official stock loss on identical noisy
   states; mean occupancy IoU > damaged input by .01.
   Shape gate: same flow-loss threshold, mean mesh CD lower than damaged input,
   exactly unchanged exterior reference features, and lower CD for the predicted
   SS rollout. Report stock local-repair comparisons as well. A gate failure
   blocks formal training rather than being renamed a success.
4. Only after both gates pass: publish the complete immutable Train2000 / Dev64
   manifests. Train fresh SS and Shape adapters for both repair and full
   reconstruction, 4,000 updates each, gradient accumulation 4 per single GPU,
   BF16, checkpointing, AdamW new-module LR 1e-4 / LoRA LR 3e-5, EMA .9995 with ramp.
   Do not initialize the formal runs from overfit weights.
5. Dev64: four interventions including identity, seeds 42/43. Compare damaged
   input, official stock local flow, native codec reconstruction, trained repair,
   and trained full multiview reconstruction. Teacher-support controls are labeled.
   Full native single-image generation and automatic alignment/localization are
   future tests, not substituted by the teacher-support stock control.

Sample 20,000 surface points per mesh; report symmetric CD-L1 and F-score at .01
and .02 in the shared normalized native frame. Report raw canonical source-mesh
GT and the VAE codec floor. No ICP, reflection, scale fitting, or GT-conditioned
policy/selection. Keep failures and empty outputs visible.

All datasets, outputs, caches, models and runtime snapshots stay on `/data/zjr`.
Source and small documentation stay in Tracker. The reused conditioning core is
snapshotted under `vendor_pose` with hashes; no runtime import of the other active
pose-constrained project's source. GPU launch waits for >=14,500 MiB free and
utilization <=15%; no unrelated process is killed or relocated.
