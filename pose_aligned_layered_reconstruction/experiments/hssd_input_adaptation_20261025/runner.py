"""One explicitly selected GPU; serial stages, artifact resume and live status."""
import argparse
import fcntl
import signal
import subprocess
import time
from collections import deque
# Direct-script entry keeps status/watch independent of the package __init__,
# which imports Torch and can stall on shared-disk library reads.
if __package__:
    from .common import *
else:
    from common import *

ACTIONS=['prepare','preview','data','smoke','all','summary']

def check_gpu():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    if not gpu.isdigit():raise ValueError('HSSD_INPUT_GPU must be one physical GPU index')
    used=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(used.strip())>1024:raise RuntimeError(f'GPU {gpu} occupied ({used.strip()} MiB); choose HSSD_INPUT_GPU explicitly')

def terminate_group(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def stage(name,command,timeout):
    log=OUT/'logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();event(name,phase='starting')
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            while True:
                write(OUT/'status.json',{'time':now(),'status':'running','stage':name,
                    'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid(),'child_pid':process.pid,
                    'elapsed_seconds':time.monotonic()-started,'log':str(log)})
                code=process.poll()
                if code is not None:
                    if code:raise RuntimeError(f'{name} exited {code}; see {log}')
                    return
                if time.monotonic()-started>timeout:raise TimeoutError(f'{name} exceeded {timeout} seconds')
                time.sleep(3)
        finally:terminate_group(process)

def module(name,*args):return [PYTHON,'-m',MODULE+'.'+name,*args]

def run(action,lock_fd=None):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(lock_fd,'a') if lock_fd is not None else (OUT/'queue.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if action not in ['prepare','summary']:check_gpu()
            write(OUT/'launches'/f'{time.time_ns()}.json',{'time':now(),'action':action,
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'sources':[binding(p) for p in sorted(HERE.glob('*.py'))]+[binding(HERE/'run.sh')]})
            if action!='summary':
                stage('prepare',module('prepare'),300)
                stage('contracts',[PYTHON,'-m','unittest',MODULE+'.test_contracts','-v'],180)
                stage('crop518_images',module('data','images','--domain','crop518'),180)
            if action in ['preview','data','all']:
                frames=[0] if action=='preview' else FRAMES
                stage('render_2072',[BLENDER,'-b','-t','4','--python-exit-code','1','-P',str(SHARED_CODE/'render.py'),
                    '--','--output',str(OUT/'hires'),'--frames',*map(str,frames)],3600)
                stage('audit_2072',module('data','audit-high',*(['--preview'] if action=='preview' else [])),600)
                stage('crop2072_images',module('data','images','--domain','crop2072',*(['--preview'] if action=='preview' else [])),180)
            if action in ['data','all','smoke']:
                stage('dino_cache',module('data','cache',*(['--domain','crop518'] if action=='smoke' else [])),1800)
            if action in ['all','smoke']:
                stage('smoke_ss',module('evaluate','ss','--smoke'),1800)
                stage('smoke_mesh',module('evaluate','mesh','--smoke'),1800)
                if not read(OUT/'smoke.json')['passed']:raise ValueError('Smoke gate failed')
            if action=='all':
                for arm in ARMS:stage('ss_'+arm,module('evaluate','ss','--arm',arm),7200)
                stage('mesh',module('evaluate','mesh'),21600)
            if action in ['all','summary']:stage('summary',module('summarize'),7200)
            write(OUT/'status.json',{'time':now(),'status':action+'_complete_review_required','stage':None,
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid()})
        except BaseException as error:
            previous=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',{**previous,'time':now(),'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                'error':str(error) or type(error).__name__})
            raise

def start(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action not in ['prepare','summary']:check_gpu()
        log=OUT/f'launcher_{time.time_ns()}.log'
        with log.open('w') as stream:
            process=subprocess.Popen([PYTHON,str(HERE/'runner.py'),action,'--lock-fd',str(lock.fileno())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        print(f'Started {action}, PID {process.pid}; log {log}',flush=True)

def status():
    s=read(OUT/'status.json') if (OUT/'status.json').exists() else {'status':'not_started'}
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(s['time'])).total_seconds() if s.get('time') else None
    print(f'[{now()}] HSSD InputAdapt4 GPU {s.get("gpu","-")}',flush=True)
    print(f'状态：{s["status"]}  阶段：{s.get("stage") or "无活动阶段"}  最近更新：{round(age) if age is not None else "-"} 秒前',flush=True)
    if s['status']=='running' and age is not None and age>30:
        print('提示：心跳已过期，请检查日志和队列 PID；不能仅凭旧 running 状态认定仍在运行。',flush=True)
    frames=sum((OUT/'hires/frames'/f'{f:03d}.json').exists() for f in FRAMES)
    caches={d:len(list((OUT/'inputs'/d).glob('*/receipt.json'))) for d in ['crop518','crop2072']}
    print(f'2072 共享帧：{frames}/4；DINO 缓存：{caches}（各 4 个）',flush=True)
    if (OUT/'plan.json').exists():
        jobs=new_jobs();ss_total=mesh_total=0
        for variant in VARIANTS[1:]:
            selected=[j for j in jobs if j[0]==variant]
            ss=sum((case_dir(v,a,o['uid'],t)/'ss.json').exists() for v,a,o,t in selected)
            mesh=sum((case_dir(v,a,o['uid'],t)/'mesh.json').exists() for v,a,o,t in selected)
            ss_total+=ss;mesh_total+=mesh
            print(f'  {variant}: 新增 SS={ss}/{len(selected)} mesh={mesh}/{len(selected)}',flush=True)
        print(f'新增合计：SS={ss_total}/54 mesh={mesh_total}/54；逻辑结果={42+mesh_total}/96（含旧引用 24 + 相同输入别名 18）',flush=True)
    smoke=read(OUT/'smoke.json') if (OUT/'smoke.json').exists() else {}
    print(f'旧基线完整重放与两视图短检查：{"通过" if smoke.get("passed") else "尚未全部通过"}',flush=True)
    if s.get('error'):print('错误：'+s['error'],flush=True)
    if s.get('log'):
        print('日志：'+s['log'],flush=True)
        if Path(s['log']).exists():
            with Path(s['log']).open(errors='replace') as stream:print(''.join(deque(stream,maxlen=2)).rstrip(),flush=True)
    if s['status']=='all_complete_review_required':print('全部推理与汇总完成；请审阅 REPORT.md、figures/ 和 world/。',flush=True)

def stop():
    s=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    if s.get('status')!='running':
        print('No running queue recorded.',flush=True);return
    pid=s.get('pid')
    cmd=Path(f'/proc/{pid}/cmdline')
    if not cmd.exists():
        write(OUT/'status.json',{**s,'time':now(),'status':'interrupted','stage':None,'error':'Queue process no longer exists'})
        return
    parts=cmd.read_bytes().split(b'\0')
    if str(HERE/'runner.py').encode() not in parts and (MODULE+'.runner').encode() not in parts:
        raise ValueError('PID no longer belongs to this queue; refusing to signal')
    os.kill(pid,signal.SIGTERM)
    print(f'Sent SIGTERM to this queue PID {pid}; it will terminate its own active child and preserve completed artifacts.',flush=True)

def signal_stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=ACTIONS+['start-'+a for a in ACTIONS]+['status','watch','stop'])
    parser.add_argument('--lock-fd',type=int);args=parser.parse_args()
    signal.signal(signal.SIGTERM,signal_stop)
    if args.action.startswith('start-'):start(args.action[6:])
    elif args.action=='stop':stop()
    elif args.action in ['status','watch']:
        while True:
            status()
            if args.action=='status':break
            time.sleep(10)
    else:run(args.action,args.lock_fd)
