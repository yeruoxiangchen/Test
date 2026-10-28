"""Serial single-GPU diagnostic queue. No training and no automatic expansion."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
from .common import OUT,MODULE,PYTHON,ADAPTERS,ARMS,read,write,now,freeze,verify


def require_preflight():
    receipt=read(OUT/'smoke.json')
    if not receipt['passed'] or receipt['protocol']!=freeze():
        raise ValueError('Missing or changed preflight')
    for row in receipt['checkpoints'].values():verify(row)
    if len(list((OUT/'audit'/'clean').glob('[0-9][0-9][0-9].json')))!=16 or len(list((OUT/'audit'/'occluded').glob('[0-9][0-9][0-9].json')))!=16:
        raise ValueError('Require full CPU LaRI audit first')
    read(OUT/'audit_summary.json')


def child(stage, command):
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/f'{stage}.log'
    with log.open('a') as f:
        f.write('\n'+now()+' '+repr(command)+'\n');f.flush()
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                write(OUT/'status.json',{'time':now(),'status':'running','stage':stage,'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),
                    'pid':os.getpid(),'child_pid':process.pid,'log':str(log)})
                time.sleep(3)
            if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        finally:
            if process.poll() is None:process.terminate();process.wait()


def queue(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if action in ['audit','preflight']:
                child(action,[PYTHON,'-u','-m',MODULE+('.audit' if action=='audit' else '.smoke')])
                state=action+'_complete'
            else:
                require_preflight()
                if action in ['ss','run']:
                    for adapter in ADAPTERS:
                        child('ss_'+adapter,[PYTHON,'-u','-m',MODULE+'.ss_probe','sample','--adapter',adapter])
                    child('ss_summary',[PYTHON,'-u','-m',MODULE+'.ss_probe','summary'])
                    state='ss_complete_review_required'
                if action in ['mesh','run']:
                    for arm in ARMS:
                        child('mesh_'+arm,[PYTHON,'-u','-m',MODULE+'.mesh_cases','decode','--arm',arm])
                    child('mesh_summary',[PYTHON,'-u','-m',MODULE+'.mesh_cases','summary'])
                    state='mesh_complete_review_required' if action=='mesh' else 'diagnostics_complete_review_required'
            write(OUT/'status.json',{'time':now(),'status':state,'stage':None,'gpu':os.getenv('CUDA_VISIBLE_DEVICES')})
        except BaseException as exc:
            old=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',{**old,'time':now(),'status':'failed','error':str(exc)})
            raise


def start(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action not in ['audit','preflight']:require_preflight()
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'launcher_{action}_{time.time_ns()}.log'
    with path.open('w') as f:
        process=subprocess.Popen(['bash',str(Path(__file__).with_name('run.sh')),action],cwd='/home/zjr/Tracker',
            start_new_session=True,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True)
    write(OUT/'launcher.json',{'pid':process.pid,'action':action,'time':now(),'log':str(path)})
    print(f'Started PID {process.pid}; log: {path}',flush=True)


def status():
    s=read(OUT/'status.json') if (OUT/'status.json').exists() else {'status':'not_started'}
    print(f'[{now()}] HSSD LaRI Causal4 GPU {s.get("gpu","-")}')
    print(f'状态：{s["status"]}  阶段：{s.get("stage") or "无活动阶段"}  最近心跳：{s.get("time","-")}')
    print('LaRI审计：'+str(len(list((OUT/'audit').glob('*/[0-9][0-9][0-9].json'))))+'/32（原16物体×两域）')
    for a in ADAPTERS:print(f'  SS {a}: {len(list((OUT/"ss"/a).glob("[0-9][0-9][0-9]/*/[0-9][0-9].json")))}/96')
    for a in ARMS:print(f'  mesh {a}: {len(list((OUT/"mesh"/a).glob("[0-9][0-9][0-9]/[0-9][0-9].json")))}/12')
    if s.get('error'):print('错误：'+s['error'])
    if s.get('log'):print('日志：'+s['log'])


def main():
    actions=['audit','preflight','ss','mesh','run']
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=actions+['start-'+a for a in actions]+['watch','status'])
    action=p.parse_args().action
    if action.startswith('start-'):start(action[6:])
    elif action in ['status','watch']:
        try:
            while True:
                status()
                if action=='status':break
                time.sleep(10)
        except KeyboardInterrupt:pass
    else:queue(action)


if __name__=='__main__':main()
