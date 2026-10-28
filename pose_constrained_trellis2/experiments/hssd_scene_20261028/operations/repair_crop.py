"""Stop only this pilot, archive receipts before correcting pixel-centre crop K."""
import os,signal,time
from pathlib import Path
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c

s=c.read(c.OUT/'status.json');pid=s['pid'];child=s.get('child_pid')
def ours(pid,token):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and token in p.read_bytes().decode(errors='replace')
if ours(pid,c.MODULE+'.runner'):
    if Path(f'/proc/{pid}/stat').read_text().split()[21]!=s['process_start']:raise RuntimeError('PID reused')
    os.kill(pid,signal.SIGTERM)
if child and ours(child,c.MODULE+'.worker'):os.kill(child,signal.SIGTERM)
for _ in range(40):
    if not ours(pid,c.MODULE+'.runner') and not (child and ours(child,c.MODULE+'.worker')):break
    time.sleep(.25)
else:raise RuntimeError('Owned processes did not exit')
dest=c.OUT/'repairs'/('pixel_centres_'+str(time.time_ns()));dest.mkdir(parents=True)
for name in ['runtime.json','status.json','stages']:
    p=c.OUT/name
    if p.exists():p.rename(dest/name)
c.write(dest/'reason.json',{'time':c.now(),'reason':'independent actual-camera audit found pixel-centre crop K offset missing; archive before correcting. Numeric stages will rerun, same12 subjects.',
                          'previous_status':s})
print('Stopped only owned Native12 queue; previous receipts preserved at',dest)
