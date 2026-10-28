# Migration contract

This directory migrates the method, not its old tensors.  The old 30K endpoint
remains a baseline and reproducibility artifact; parameter shapes and latent
semantics changed enough that direct checkpoint reuse would be misleading.
Its frozen object selection and posed RGB observations are reused as the data
protocol, while native T2 targets must be regenerated from matching raw assets.

## Kept unchanged at the method level

- Random ordered subsets of one to eight calibrated input views.
- Frustum projection of DINO spatial tokens onto native 3D query locations.
- The learned IBRNet-style mean/variance view aggregator.
- An independent zero-initialized condition projection before every Flow block.
- Attention LoRA with a frozen official backbone.
- Conditional/unconditional flow matching, EMA, and separate SS/Shape training.
- Actual sparse active coordinates as Shape-stage projection queries.

The projection equation is unchanged at the method level, but its camera-axis
contract is now explicit.  In particular, ProObjaverse's OpenGL/Blender
camera `+Y` is flipped when mapped to downward image rows.  This repairs a
latent vertical-axis omission in the previous cache/projection path; generic
OpenCV inputs retain `camera_image_y_sign=+1`.

## Replaced for TRELLIS.2

| Previous backend concept | TRELLIS.2 replacement |
| --- | --- |
| DINO patches at the previous encoder resolution | DINOv3 ViT-L/16 at 512, 1029 full tokens |
| TRELLIS SS Flow | T2 RoPE SS Flow, 8 channels at 16³, 1536 width, 30 blocks |
| TRELLIS SLat Flow | T2 RoPE Shape-SLat-512 Flow, 32 channels at active 32³ coordinates |
| Stock mesh VAE decoder | T2 Flexible Dual Grid geometry decoder at 512 |
| Previous SS/SLat latent caches | Native T2 SS and Shape encoder targets |
| Previous model checkpoint initialization | Official T2 Flow checkpoints only |

## Deliberate 2D-context choice

TRELLIS.2's released image-conditioned geometry Flows are trained with one
DINOv3 image sequence.  The default migration therefore sends the first
selected view's full 1029 tokens through ordinary cross-attention, while the
pose branch projects and aggregates every selected view.  This isolates the
new multiview information to the pose-aware mechanism at initialization and
retains the official single-view context distribution.  Concatenating all
selected sequences is available only through `--context-mode concat` and must
be reported as a separate ablation.

The old Shape-stage implementation could pass a list of per-view contexts to a
TRELLIS-v1 block with project-specific fusion semantics. That call surface is
not a native TRELLIS.2 checkpoint contract. Keeping one complete reference
sequence in ordinary cross-attention and placing every selected view in the
pose-projected branch is therefore the controlled migration; it is not copied
from ReconViaGen's TRELLIS.2 velocity-fusion implementation.

## Explicit exclusions

- No texture Flow, texture latent, PBR decoder, or appearance loss.
- No AR world-frame deployment or application-specific mesh refinement.
- No import from either old model package.
- No GenRecon runtime dependency.
- No distillation and no loading of either previous 30K checkpoint.
- No optional TRELLIS.2 1024 Shape cascade in this first geometry migration.

These exclusions make the first experiment answer one narrow question: whether
the same pose-constrained multiview mechanism transfers cleanly to the released
TRELLIS.2 512 geometry generation stack.
