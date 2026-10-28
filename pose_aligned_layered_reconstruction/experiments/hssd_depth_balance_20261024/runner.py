"""One fixed GPU, serial stages, strict resume, and explicit stop before mesh."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from .common import ARMS, REFERENCES, OUT, MODULE, PYTHON, LARI_STEPS, SS_STEPS, freeze, read, verify, write, now


def require_preflight():
    smoke=read(OUT/'smoke.json')
    if not smoke['passed'] or smoke['protocol']!=freeze():
        raise ValueError('Missing/changed preflight')
    for value in smoke['evidence']:verify(value)


def validate_training(kind):
    steps=LARI_STEPS if kind=='lari' else SS_STEPS
    runs=[]
    for arm in ARMS:
        root=OUT/'train'/f'{kind}_{arm}'
        run,done=read(root/'run.json'),read(root/'complete.json')
        if done['step']!=steps or done['contract']!=run['contract'] or run['contract']['protocol']!=freeze():
            raise ValueError('Incomplete or changed training')
        verify(done['checkpoint'])
        history=[__import__('json').loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
        if [r['step'] for r in history]!=list(range(1,steps+1)):
            raise ValueError('Training history cursor differs')
        runs.append((run,[(r['uid'],r['learning_rate'],r.get('diffusion_time')) for r in history]))
    if runs[0][0]['initial_state_sha']!=runs[1][0]['initial_state_sha'] or runs[0][0]['trainable_parameter_count']!=runs[1][0]['trainable_parameter_count'] or runs[0][1]!=runs[1][1]:
        raise ValueError('Unmatched initialization/capacity/order/schedule/noise time')


def child(stage, command):
    log=OUT/'logs'/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        stream.write(__import__('json').dumps({'time':now(),'command':command})+'\n');stream.flush()
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                write(OUT/'status.json',{'time':now(),'status':'running','stage':stage,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'pid':os.getpid(),'child_pid':process.pid,'log':str(log)})
                time.sleep(3)
            if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        finally:
            if process.poll() is None:process.terminate();process.wait()


def command(module,*args):
    return [PYTHON,'-u','-m',MODULE+'.'+module,*args]


def queue(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if action=='preflight':
                child('preflight',command('smoke'));state='preflight_complete'
            else:
                require_preflight()
                if action in ('train','run'):
                    for arm in ARMS:child('lari_'+arm,command('train','lari','--arm',arm))
                    validate_training('lari')
                    child('lari_audit',command('audit'))
                    for arm in ARMS:child('ss_train_'+arm,command('train','ss','--arm',arm))
                    validate_training('ss');state='training_complete_pending_ss'
                if action in ('ss','run','mesh'):
                    validate_training('lari');validate_training('ss')
                    if action=='mesh':
                        for arm in (*REFERENCES,*ARMS):child('mesh_'+arm,command('evaluate','mesh','--arm',arm))
                        child('mesh_summary',command('evaluate','mesh-summary'));state='mesh_complete_review_required'
                    else:
                        for arm in (*REFERENCES,*ARMS):child('ss_'+arm,command('evaluate','ss','--arm',arm))
                        child('ss_summary',command('evaluate','ss-summary'));state='ss_complete_review_required'
            write(OUT/'status.json',{'time':now(),'status':state,'stage':None,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as error:
            path=OUT/'status.json';old=read(path) if path.exists() else {}
            write(path,{**old,'time':now(),'status':'failed','error':str(error)})
            raise


def start(action):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action!='preflight':require_preflight()
        if action in ('ss','mesh'):validate_training('lari');validate_training('ss')
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    log=logs/f'launcher_{action}_{time.time_ns()}.log'
    with log.open('w') as stream:
        process=subprocess.Popen(['bash',str(Path(__file__).with_name('run.sh')),action],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    write(OUT/'launcher.json',{'time':now(),'pid':process.pid,'action':action,'log':str(log)})
    print(f'Started PID {process.pid}; log: {log}',flush=True)


def status():
    state=read(OUT/'status.json') if (OUT/'status.json').exists() else {'status':'not_started'}
    print(f'[{now()}] HSSD DepthBalance16 GPU {state.get("gpu","-")}')
    print(f'状态：{state["status"]}  阶段：{state.get("stage") or "无活动阶段"}  最近心跳：{state.get("time","-")}')
    if state.get('error'):print('错误：'+state['error'])
    for arm in ARMS:
        progress=[]
        for kind,total in [('lari',LARI_STEPS),('ss',SS_STEPS)]:
            file=OUT/'train'/f'{kind}_{arm}'/'history.jsonl';last={}
            if file.exists():
                for line in reversed(file.read_text().splitlines()):
                    try:last=__import__('json').loads(line);break
                    except ValueError:continue
            progress.append(f'{kind}={last.get("step",0)}/{total}')
        ns=len(list((OUT/'evaluations'/arm).glob('[0-9][0-9][0-9]/*_ss.json')))
        nm=len(list((OUT/'mesh'/arm).glob('[0-9][0-9][0-9]/[0-9][0-9].json')))
        print(f'  {arm}: '+ ' '.join(progress)+f' SS={ns}/48 mesh={nm}/12')
    if state.get('log'):print('日志：'+state['log'])
    if state['status']=='ss_complete_review_required':print('SS已完成，等待审阅；mesh尚未自动启动。')
    if state['status']=='preflight_complete':print('仅短时预检完成，正式训练尚未启动。')


if __name__=='__main__':
    actions=['preflight','train','ss','mesh','run']
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=actions+['start-'+a for a in actions]+['status','watch'])
    a=p.parse_args().action
    if a.startswith('start-'):start(a[6:])
    elif a in ('status','watch'):
        while True:
            status()
            if a=='status':break
            time.sleep(10)
    else:queue(a)
