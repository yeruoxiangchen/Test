# TRELLIS.2 geometry repair

Native official TRELLIS.2 Shape-512 generation and input-grounded local editing
of its final Shape SLat. Base weights and the geometry decoder are frozen.
There is no texture model, old30K checkpoint, or pose-adapter training in this
path. The upstream TRELLIS2 checkout remains unchanged.

Source lives here; assets, inputs, latents, OBJ meshes, figures, logs and all
runtime caches live under `/data/zjr/trellis2_geometry_repair`.

```bash
bash trellis2_geometry_repair/run.sh assets
bash trellis2_geometry_repair/run.sh prepare
bash trellis2_geometry_repair/run.sh tests
bash trellis2_geometry_repair/run.sh launch trellis2-repair-pilot-20261028 --gpu 6
watch -n 10 -d 'bash /home/zjr/Tracker/trellis2_geometry_repair/run.sh status'
```

See [PROTOCOL.md](PROTOCOL.md), [VALIDATION.md](VALIDATION.md), and [审计.md](审计.md). The initial pilot uses
four previously inspected development objects, two seeds, one reference image
for stock generation and posed multiview images for registration and repair.
It is a backend/repair pilot, not an independent generalization benchmark.
