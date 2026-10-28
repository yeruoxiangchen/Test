import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .config import OUT, PARAMS, folder, registry, write, read, now

REPO=Path('/home/zjr/Tracker')
LAUNCHER=REPO/'visual_evidence_probe/run_failure_repair.sh'


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=OUT/'pipeline.json'
    if state_path.exists():
        write(OUT/'pipeline_history'/f'{now().replace(":","-")}.json',read(state_path))
    state=dict(started=now(),updated=now(),gpu=a.gpu,status='running',attempts=[],completed_stages=[])
    write(state_path,state)
    def stage(name,indices=None,extra=()):
        stamp=now().replace(':','-');log=logs/f'{name}_{stamp}.log'
        command=['bash',str(LAUNCHER),name]
        if name in ('generate','regions','depth','run','evaluate','figures'):
            command+=['--gpu',a.gpu]
        if indices is not None:command+=['--indices',','.join(map(str,indices))]
        command+=list(extra)
        attempt=dict(stage=name,indices=indices,started=now(),log=str(log))
        state.update(stage=name,active_log=str(log),updated=now());state['attempts'].append(attempt);write(state_path,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_LAUNCH_BLOCKING='1')
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state.update(child_pid=process.pid);write(state_path,state)
            while True:
                try:code=process.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    state.update(updated=now());write(state_path,state)
        attempt.update(returncode=code,finished=now());state.update(updated=now());write(state_path,state)
        if code:raise RuntimeError(f'{name} failed: {log}')
        state['completed_stages'].append(dict(stage=name,indices=indices,finished=now()));write(state_path,state)
    def cohort(indices,smoke=False):
        if not indices:return
        stage('generate',indices);stage('regions',indices);stage('depth',indices)
        stage('run',indices);stage('evaluate',indices);stage('report');stage('figures',indices)
        stage('check',indices,('--smoke',) if smoke else ())
    try:
        if not (OUT/'preparation_complete.json').exists():stage('prepare')
        indices=[r['index'] for r in registry()['rows']]
        if not (OUT/'historical_analysis.json').exists():stage('report',extra=('--historical',))
        if not (OUT/'smoke_complete.json').exists():cohort(indices[:1],smoke=True)
        pending=[i for i in indices if any(not (folder(i,s)/'complete.json').exists() for s in PARAMS['seeds'])]
        cohort(pending)
        stage('check',indices);stage('report');stage('figures',indices);stage('closeout')
        state.update(status='complete',stage='complete',updated=now());write(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(state_path,state);raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
