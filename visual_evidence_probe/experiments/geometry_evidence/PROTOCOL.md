# Sixth probe: real geometric error and input evidence

2026-10-27. Same eight previously inspected Omni scanned objects x two seeds.
Frozen original30K SS/SLat endpoints and mesh decoder; no model training.
All runtime files and caches are under /data/zjr/visual_evidence_probe.

The four arms use exactly the same original latent, support, source-only
editable rows, image-derived local region, posed training views, optimizer,
80-step budget, feature bound0.5 training std, and exterior protection.
The larger bound/budget is an explicit amendment to round five, not a paired
comparison to the old0.12/40-step results. Checkpoints0/20/40/80 are retained.

A: dense correct camera-z and normals in the input-derived region.
A-depth: same dense correct camera-z with local normals disabled. This
auxiliary arm separates depth density from extra normal supervision.
B: GT camera-z queried at exactly the saved input-matcher pixel identities.
C: the original input-estimated camera-z at those same pixels. No estimated
local normals are invented. All arms use observed input silhouettes, baseline
exterior geometry, and common fixed loss weights. B and C have no local normal
loss. A-depth versus B isolates available depth density; B versus C addresses
accuracy subject to the GT-intersection qualification below.

The input pool, local region, source points, editable rows, cameras, and input
targets are saved and sealed BEFORE GT is loaded. Existing fifth-round pools
are reused, not recomputed or filtered by GT. When a pool lacks a region,
the already-saved fourth-round input-selected face mask defines it. Such cases
remain no-operation for B/C if they have no queried pixels; A/A-depth still run.
An oracle query with no GT intersection supplies no positive surface depth;
it is counted explicitly and never filled with a fake zero depth. B/C query
identities remain equal but their valid-depth counts can differ. Only cases
with all queried rays intersecting GT support a pure paired-depth-quality
interpretation. All cases remain in the overall report.

Every optimizer selects checkpoints using its OWN training objective, never
evaluation metrics. A/B selection is privileged diagnostic supervision. C
checkpoint selection and optional acceptance are input-only. Report fixed
final-step proposal, training-selected result, and accepted result separately;
do not interpret an oracle arm as an implementable method. Input acceptance
uses the existing heldout photometric evidence and common exterior/topology
checks. Missing validation is unable, not an accepted edit.
The inherited silhouette gate bounds change FROM THE ORIGINAL mesh (IoU>=.995),
not regression against the observed input mask; it can reject beneficial
silhouette correction. It checks the own-training-selected candidate only and
does not reproduce round five's gate-based search over every checkpoint.
Accepted rates are therefore not directly paired to round five. This explicit
interpretation does not alter any numerical optimization or saved decision.

Eight deterministic new cameras are generated from input camera distance and
fixed origin, without looking at GT or arm endpoints. They are at least10deg
from the original input directions, use identical K/W for every mesh, and
enter neither optimization nor selection. They are new evaluation viewpoints
of already-inspected objects, not an independent-object/real-capture dataset.
No new alignment, ICP, or per-arm normalization is allowed.

Evaluation includes fixed-domain local camera-z error on input and new views,
surface CD/F-score, silhouette, area outside the edit, components, nonmanifold
edges, original-surface displacement, changed area, time and peak GPU memory.
The local region is fixed from the original mesh; missing predicted rays pay
0.1 with a fixed denominator. Report target-only and original-union domains,
valid-ray counts, all no-op/failure cases, paired active-only subsets and
object-level (not seed/pixel-level) bootstrap intervals. GT does not define the
input region or move the cameras. Previous experiments and audit text remain.
