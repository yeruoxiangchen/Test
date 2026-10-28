#!/usr/bin/env bash
# Prepare inputs, then automatically select any four idle GPUs and run once.
# Use systemd-run to keep the queue alive after disconnecting SSH.
# Add --allow-shared-gpus to accept <=2048 MiB use when free memory also fits.
set -euo pipefail
SCA_QUEUE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/bash "$SCA_QUEUE_ROOT/scripts/run_repair_workflow.sh" all \
  --gpus auto --wait-for-gpus --resume-interrupted "$@"
