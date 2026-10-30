#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/zjr/pose_constrained_trellis2/packages:/home/zjr/Tracker:/home/zjr/Tracker/TRELLIS2
PY=/home/zjr/anaconda3/envs/trellis2/bin/python
MODULE=pose_constrained_trellis2.experiments.hssd_relations192_20261029
case "${1:-status}" in
  prepare) exec "$PY" -B -m "$MODULE.prepare" ;;
  test) exec "$PY" -B -m "$MODULE.tests" ;;
  start) exec "$PY" -B -m "$MODULE.runner" start --mode all ;;
  start-preflight) exec "$PY" -B -m "$MODULE.runner" start --mode preflight ;;
  start-support-check) exec "$PY" -B -m "$MODULE.runner" start --mode support-check ;;
  start-data) exec "$PY" -B -m "$MODULE.runner" start --mode data ;;
  start-train) exec "$PY" -B -m "$MODULE.runner" start --mode train ;;
  start-eval) exec "$PY" -B -m "$MODULE.runner" start --mode eval ;;
  start-test) exec "$PY" -B -m "$MODULE.runner" start --mode test ;;
  status|watch|stop) exec "$PY" -B -m "$MODULE.runner" "${1:-status}" ;;
  *) echo 'run.sh prepare|test|start-preflight|start-support-check|start-data|start-train|start-eval|start-test|start|status|watch|stop'; exit 2 ;;
esac
