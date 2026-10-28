# Round eight: constrained local evidence editing

Frozen on 2026-10-28, after examining round seven. These eight Omni objects
are development data, not an independent validation set. All old round-seven
results remain read-only. Runtime data and caches live under /data/zjr.

Three arms: the exact existing dense+observed-policy reference; actual decoded
mesh surface anchors plus a train-view/3D trust region; the same constrained
editor with frozen input-only teacher-depth reliability weights. The last two
share original SS, editable rows, regions, cameras, targets, frozen30K weights,
80 attempted Adam updates, feature bound0.5std, and the round-seven final gate.
The anchored editor is a joint intervention (3D anchors, normal preservation,
and five-step trust rollback); it does not isolate each contribution.

1024 original exterior surface anchors are area-sampled without GT. The loss
uses closest points on the16 centroid-neighbor current triangles, not nearest
vertices or corner-SDF. This is an approximate surface search, not a certified
global triangle minimum. Exterior normal changes are penalized. Every five
attempted updates, require 3D anchor displacement mean<=0.25/256 and p95<=1/256,
and the same train-view exterior depth limits. A violating group restores the
last feasible latent AND Adam moments and halves learning rate to a fixed floor.
Report attempted updates, rollbacks, saved checkpoints and actual time. Fewer
effective updates are possible; do not claim equal accepted-update count.

The reliability arm preserves all NCC anchors at unit weight. Additional DA3
rays are checked by training-view patch cost and their agreement with raw
teacher depth at NCC anchors. Contradicting textured rays get zero weight;
unverifiable textureless rays retain weak0.1 weight times agreement. This is a
heuristic confidence estimate, not a calibrated correctness probability.
Heldout RGB never enters reliability or gradient/trust-region decisions.

Optimization and selection load no GT. Each case seals every input-only arm
before a separate evaluator uses the existing GT and same eight novel cameras.
Report accepted AND own-training-selected candidates, fixed80-step proposals,
CD/F-score, fixed local depth, exterior drift, anchor drift, acceptance and time.
Seeds are averaged within objects. A development pilot uses4,5,6,10/seed42;
then the same frozen configuration may run the eight development objects x2.
Only after demonstrable benefit should a fresh unseen-by-this-development
holdout cohort be frozen; the previous16 validation objects have now informed
design and must not be relabeled as independent validation of round eight.

The BOP input audit is a separate diagnostic using annotated geometry and
sensor depth. It neither modifies the original inputs nor supplies GT depth,
alignment, weights, targets or acceptance decisions to the editor.

Implementation correction before formal development completion: on2026-10-28
the first actual consecutive rollback (010/42) exposed same-device Adam state
aliasing on load_state_dict. The entire affected debugging case is retained in
debug_optimizer_alias_v1 and rerun. Other completed cases had zero rollbacks
and are unaffected. The corrected loader deep-copies the backup on every
restore; a repeated-rollback regression verifies immutable backup moments and
step counters. Scientific hyperparameters and method decisions are unchanged.
