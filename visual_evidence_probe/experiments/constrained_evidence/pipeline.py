import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from .config import OUT,PARAMS,folder,read,write,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--wait-unit');p.add_argument('--formal',action='store_true');p.add_argument('--evaluation-only',action='store_true');p.add_argument('--indices');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    lock=(OUT/f'pipeline_gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(time=now(),updated=now(),gpu=a.gpu,status='running',attempts=[],formal=a.formal)
    receipt=OUT/f'pipeline_gpu{a.gpu}.json';write(receipt,state)
    if a.wait_unit:
        if not a.wait_unit.startswith('visual-evidence-constrained-'):raise ValueError('invalid wait unit')
        state['stage']='waiting_for_smoke';write(receipt,state)
        while subprocess.run(['systemctl','--user','is-active','--quiet',a.wait_unit]).returncode==0:time.sleep(15)
    def stage(name,indices=None,seeds=None,blocking=False):
        log=OUT/'logs'/f'pipeline_{a.gpu}_{name}_{now().replace(":","-")}.log'
        cmd=[sys.executable,'-B','-u','-m',f'visual_evidence_probe.experiments.constrained_evidence.{name}']
        if indices is not None:cmd+=['--gpu',a.gpu,'--indices',','.join(map(str,indices)),'--seeds',','.join(map(str,seeds))]
        row=dict(stage=name,indices=indices,seeds=seeds,started=now(),log=str(log),blocking=blocking)
        state.update(stage=name,active_log=str(log),updated=now());state['attempts'].append(row);write(receipt,state)
        env=dict(os.environ,CUDA_LAUNCH_BLOCKING='1' if blocking else '0')
        with log.open('a') as stream:result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,finished=now());state['updated']=now();write(receipt,state)
        if result.returncode:
            error=RuntimeError(f'{name} failed: {log}');error.cuda_error='CUDA error' in log.read_text();raise error
    def batch(indices,seeds):
        pending=[i for i in indices if any(not (folder(i,s)/'input_complete.json').exists() for s in seeds)]
        if pending and a.evaluation_only:raise ValueError('input decisions not sealed for evaluation-only task')
        if pending:
            try:stage('run',pending,seeds)
            except RuntimeError as error:
                if not getattr(error,'cuda_error',False):raise
                stage('run',pending,seeds,blocking=True)
        pending=[i for i in indices if any(not (folder(i,s)/'complete.json').exists() for s in seeds)]
        if pending:stage('evaluate',pending,seeds)
        stage('report');stage('figures',indices,seeds)
    try:
        if a.evaluation_only:
            batch(list(map(int,a.indices.split(','))),list(map(int,a.seeds.split(','))))
            state.update(status='subset_complete',stage='complete',updated=now());write(receipt,state);return
        batch(PARAMS['pilot_indices'],[42])
        state['pilot_complete']=True;write(receipt,state)
        if a.formal:
            batch(PARAMS['dev_indices'],PARAMS['seeds']);stage('closeout')
        state.update(status='complete' if a.formal else 'pilot_complete',stage='complete',updated=now());write(receipt,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(receipt,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
