#!/usr/bin/env bash
set -euo pipefail
cd /home/zjr/Tracker
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/zjr/pose_constrained_trellis2/packages:/home/zjr/Tracker:/home/zjr/Tracker/TRELLIS2
PY=/home/zjr/anaconda3/envs/trellis2/bin/python
MODULE=pose_constrained_trellis2.experiments.hssd_scene_20261028
case "${1:-status}" in
  setup) exec "$PY" -B -m "$MODULE.setup" ;;
  refreeze-empty)
    "$PY" -B - <<'PY'
import fcntl,time
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
with (c.OUT/'queue.lock').open('a') as q:
    fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if list((c.OUT/'stages').glob('*.json')):raise SystemExit('已存在完成阶段，拒绝无条件重冻结')
    dest=c.OUT/'repairs'/str(time.time_ns());dest.mkdir(parents=True)
    for name in ['runtime.json','status.json']:
        p=c.OUT/name
        if p.exists():p.rename(dest/name)
    c.write(dest/'reason.json',{'reason':'CPU dual-grid importer requires visible CUDA driver; no numerical stage completed'})
PY
    ;;
  start|start-prepare)
    "$PY" -B - <<'PY'
import fcntl,os,subprocess
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
c.OUT.mkdir(parents=True,exist_ok=True)
with (c.OUT/'launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (c.OUT/'queue.lock').open('a') as q:
        try:fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('已有队列运行；请使用 watch')
    env=c.environment()
    with (c.OUT/'launcher.log').open('a') as log:
        p=subprocess.Popen([c.PYTHON,'-B','-m',c.MODULE+'.runner','run'],cwd=c.REPO,env=env,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    print('已启动原生迁移验收队列，PID',p.pid,'；默认GPU7，仅空卡进入。')
PY
    ;;
  status|watch) exec "$PY" -B -m "$MODULE.runner" "$1" ;;
  *) echo "用法：run.sh setup|start|status|watch"; exit 2 ;;
esac
