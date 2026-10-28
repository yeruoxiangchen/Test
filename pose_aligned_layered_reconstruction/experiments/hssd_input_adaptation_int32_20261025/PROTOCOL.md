# InputAdapt4 int32 runtime amendment

The original experiment, sources, inputs, seeds, weights and metrics are frozen.
Keep all 54 predicted SS supports and the 50 completed mesh records unchanged.
Finish the four recorded missing slots without cropping points or spatial grids,
changing channels, numerical precision, decoder resolution, or pose alignment.

The failed bench has 105283 support points. Its second subdivision produces
6738112x192 fp16 input features (2587435008 bytes), beyond the compiled Native
ShuffleAC GEMM's signed-int32 byte address range. More free VRAM does not remove
that address constraint. Restrict the workaround to native fp16 inference for the
192->96, 3x3x3 submanifold convolution, whose output remains addressable.

For each non-center kernel offset, retain every original indice pair. Partition
input rows into contiguous blocks of 524288, rebase only their input indices and
retain global output indices. Within an offset each output receives at most one
source; preserve original offset order and beta=1 accumulation. Retain the exact
full center torch.mm, KRSC filter slice strides, fixed native descriptor, bias and
activation. Use the original native path when the input is below the byte limit.
Never disable the native guard, patch site-packages, or switch backend globally.

Before accepting amended results require bit-exact numerical checks:

1. Holey shuffled synthetic grids, cross-block neighbors, bias on/off, two block
   sizes: coordinates and features equal to FixedNative.
2. Archived chair SLat: forced row tiling, actual convolution features/coordinates,
   raw decoder vertices/faces/attributes, and exported mesh equal to original.
3. Archived 85267-point bench seed42, close to the limit: full SLat replay, actual
   convolution and mesh checks as above, production block size 524288.

Only then resume the missing bench seed44 and stool seeds42/43/44 serially on one
explicit GPU. Save exact sampled SLat with input/SS/weight/runtime bindings before
decoding so interruption does not require resampling. Do not treat these caches as
new independently generated data. All original sources/results are hash-verified.

The original semantic identity remains in each result, with a separately hashed
runtime_amendment and regression receipts. This is an execution revision, not an
assertion that the original runtime was untouched. Do not mix results if any exact
regression fails. Original logs, including both CUDA failures, remain available.

Final summary requires all 54 new meshes (96 logical slots include 24 references
and 18 aliases), all gates, unchanged 50 original meshes, and explicit amendment
provenance. Keep the original metric/gallery/world assembly code and no fitted
transform. The bench's overprediction remains a model diagnostic, irrespective
of whether the resource-limit repair succeeds. This four-object, one-scene test
does not establish held-out generalization or LaRI effectiveness.

Long stages run separately from the interactive coding session and expose status,
progress and logs. The user monitors them. Short tests are run by Codex; do not
announce mesh/summary completion until corresponding artifacts actually exist.
