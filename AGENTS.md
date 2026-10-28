# Workspace I/O policy

User preference recorded on 2026-10-26:

- Keep datasets, checkpoints, experiment outputs, rendered meshes/images/videos,
  temporary data, and runtime/model caches under `/data/zjr/<project>/`.
- Keep Tracker/home for source code and small documentation. Do not create new
  large outputs or repeatedly scan/hash large files on the home filesystem.
- For old hardcoded paths, use compatibility links or explicit path mappings;
  preserve frozen protocols and record physical storage separately.
- Configure `TMPDIR`, `TORCH_HOME`, `HF_HOME`, `HF_HUB_CACHE`,
  `TORCH_EXTENSIONS_DIR`, and `MPLCONFIGDIR` to project directories on `/data/zjr`
  before launching new work. Avoid home bytecode/cache writes where practical.
- Do not relocate unrelated active jobs or alter shared caches without checking
  their use. Migrate the current task's data at low I/O priority and verify the
  transfer before removing its home copy.
