"""Serial one-GPU queue; resume completed artifacts and expose current counts."""
import argparse
import fcntl
import os
import subprocess
import time
from .common import *


def stage(name, command, timeout=14400):
    log=OUT/'logs'/f'{name}_{time.time_ns()}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    event(name,phase='starting',command=command)
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                write(OUT/'status.json',{'time':now(),'status':'running','stage':name,
                    'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid(),
                    'child_pid':process.pid,'elapsed_seconds':time.monotonic()-started,'log':str(log)})
                if time.monotonic()-started>timeout:raise TimeoutError(name+' exceeded '+str(timeout)+' seconds')
                time.sleep(3)
            if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; see {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def run(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Capture launched code versions separately from the immutable data plan.
            write(OUT/'launches'/f'{time.time_ns()}.json',{'time':now(),'action':action,
                'sources':[binding(p) for p in sorted(HERE.glob('*.py'))]+[binding(HERE/'run.sh')]})
            if action in ['data','baseline','all','smoke','preview']:
                stage('prepare',[PYTHON,'-m',MODULE+'.prepare'],timeout=180)
            if action in ['data','all','smoke','preview']:
                command=[str(BLENDER),'-b','-t','4','--python-exit-code','1','-P',str(HERE/'render.py'),
                         '--','--output',str(OUT)]
                if action=='smoke':command+=['--frames',*map(str,FRAMES)]
                if action=='preview':command+=['--frames','0']
                stage('render_shared_rgb',command,timeout=3600)
                if action!='preview':
                    stage('cache_shared_inputs',[PYTHON,'-m',MODULE+'.cache'],timeout=1800)
                    stage('export_gena_inputs',[PYTHON,'-m',MODULE+'.external','export'],timeout=180)
            if action in ['baseline','all','smoke']:
                for arm in ARMS[:1] if action=='smoke' else ARMS:
                    command=[PYTHON,'-m',MODULE+'.evaluate','ss','--arm',arm]
                    if action=='smoke':command+=['--limit','1']
                    stage('ss_'+arm,command,timeout=3600)
                command=[PYTHON,'-m',MODULE+'.evaluate','mesh']
                if action=='smoke':command+=['--limit','1']
                stage('mesh',command,timeout=7200)
                if action!='smoke':stage('summary',[PYTHON,'-m',MODULE+'.evaluate','summary'],timeout=1200)
            if action=='gena-download':stage('gena_download',[PYTHON,'-m',MODULE+'.external','download'],timeout=43200)
            if action=='gena-smoke':stage('gena_official_smoke',[PYTHON,'-m',MODULE+'.external','smoke'],timeout=1800)
            write(OUT/'status.json',{'time':now(),'status':action+'_complete_review_required','stage':None,
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as error:
            previous=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',{**previous,'time':now(),
                'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                'stage':None if isinstance(error,KeyboardInterrupt) else previous.get('stage'),
                'error':str(error) or type(error).__name__})
            raise


def start(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if action!='gena-download':
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','3')
        if not gpu.isdigit():raise ValueError('Use one physical GPU index')
        text=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(text.strip())>1024:raise RuntimeError(f'GPU {gpu} is occupied ({text.strip()} MiB); choose HSSD_SHARED_GPU explicitly')
    log=OUT/f'launcher_{time.time_ns()}.log'
    with log.open('w') as stream:
        process=subprocess.Popen(['bash',str(HERE/'run.sh'),action],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(f'Started {action}, PID {process.pid}; log {log}',flush=True)


def status():
    s=read(OUT/'status.json') if (OUT/'status.json').exists() else {'status':'not_started'}
    print(f'[{now()}] HSSD SharedScene4 GPU {s.get("gpu","-")}',flush=True)
    print('状态：'+s['status']+'  阶段：'+(s.get('stage') or '无活动阶段'),flush=True)
    print('共享照片：'+str(len(list((OUT/'frames').glob('*.json'))))+'/12；输入缓存：'+str(len(list((OUT/'objects').glob('*/receipt.json'))))+'/4',flush=True)
    for arm in ARMS:
        root=OUT/'evaluations'/arm
        print(f'  {arm}: SS={len(list(root.glob("*/*/ss.json")))}/12 mesh={len(list(root.glob("*/*/mesh.json")))}/12',flush=True)
    if s.get('error'):print('错误：'+s['error'],flush=True)
    if s.get('note'):print('说明：'+s['note'],flush=True)
    if s.get('log'):
        print('日志：'+s['log'],flush=True)
        if Path(s['log']).exists():
            print('\n'.join(Path(s['log']).read_text(errors='replace').splitlines()[-2:]),flush=True)


if __name__=='__main__':
    actions=['data','baseline','all','smoke','preview','gena-download','gena-smoke']
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=actions+['start-'+a for a in actions]+['status','watch'])
    a=p.parse_args().action
    if a.startswith('start-'):start(a[6:])
    elif a in ['status','watch']:
        while True:
            status()
            if a=='status':break
            time.sleep(10)
    else:run(a)
