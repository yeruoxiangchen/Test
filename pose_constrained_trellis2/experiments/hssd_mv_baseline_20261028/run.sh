#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/zjr/pose_constrained_trellis2/packages:/home/zjr/Tracker:/home/zjr/Tracker/TRELLIS2
PY=/home/zjr/anaconda3/envs/trellis2/bin/python
MODULE=pose_constrained_trellis2.experiments.hssd_mv_baseline_20261028
case "${1:-status}" in
  prepare) exec "$PY" -B -m "$MODULE.prepare" ;;
  test) exec "$PY" -B -m "$MODULE.tests" ;;
  start-preflight) exec "$PY" -B -m "$MODULE.runner" start --mode preflight ;;
  start) exec "$PY" -B -m "$MODULE.runner" start --mode all ;;
  status|watch|stop|refreeze-empty) exec "$PY" -B -m "$MODULE.runner" "${1:-status}" ;;
  *) echo '用法：run.sh prepare|test|start-preflight|start|status|watch|stop';exit 2 ;;
esac
