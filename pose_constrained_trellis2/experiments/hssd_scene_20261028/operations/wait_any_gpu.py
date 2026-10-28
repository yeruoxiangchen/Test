"""Expand only an idle waiter's allowed GPU list; preserve all numerical stages."""
import os,signal,time,subprocess
from pathlib import Path
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
s=c.read(c.OUT/'status.json');pid=s['pid'];p=Path(f'/proc/{pid}')
if s['status']!='waiting_for_gpu':raise SystemExit('Not waiting; no process changed')
if c.MODULE+'.runner' not in (p/'cmdline').read_bytes().decode():raise RuntimeError('Wrong PID')
if (p/'stat').read_text().split()[21]!=s['process_start']:raise RuntimeError('PID reused')
os.kill(pid,signal.SIGSTOP)
latest=c.read(c.OUT/'status.json')
if latest['status']!='waiting_for_gpu':
    os.kill(pid,signal.SIGCONT);raise SystemExit('Stage started; left untouched')
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(40):
    if not (p/'cmdline').exists() or not (p/'cmdline').read_bytes():break
    time.sleep(.25)
else:raise RuntimeError('Waiter did not exit')
env=c.environment();env['T2_SCENE_GPUS']='0,6,7'
with (c.OUT/'launcher.log').open('a') as log:
    child=subprocess.Popen([c.PYTHON,'-B','-m',c.MODULE+'.runner','run'],cwd=c.REPO,env=env,
                          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
c.write(c.OUT/'gpu_wait_change.json',{'time':c.now(),'old_pid':pid,'new_pid':child.pid,'gpus':[0,6,7],
    'admission':'unchanged: used<512MiB, free>=22000MiB, utilization0; no stage interrupted'})
print('Resumed waiting queue',child.pid,'allowed idle GPUs 0,6,7')
