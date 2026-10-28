import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from .config import OUT,PARAMS,folder,read,write,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--wait-unit');p.add_argument('--evaluation-only',action='store_true');p.add_argument('--indices');p.add_argument('--seeds',default='42,43');p.add_argument('--closeout-only',action='store_true');p.add_argument('--blocking',action='store_true');a=p.parse_args()
    lock=(OUT/f'pipeline_gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(started=now(),updated=now(),gpu=a.gpu,status='running',attempts=[],privileged_diagnostic=True);receipt=OUT/f'pipeline_gpu{a.gpu}.json'
    if receipt.exists():write(OUT/'pipeline_history'/f'gpu{a.gpu}_{now().replace(":","-")}.json',read(receipt))
    write(receipt,state)
    if a.wait_unit:
        if not a.wait_unit.startswith('visual-evidence-paired-'):raise ValueError('invalid wait unit')
        state['stage']='waiting_for_smoke';write(receipt,state)
        while subprocess.run(['systemctl','--user','is-active','--quiet',a.wait_unit]).returncode==0:time.sleep(15)
    def stage(name,indices=None,seeds=None,blocking=False):
        log=OUT/'logs'/f'pipeline_{a.gpu}_{name}_{now().replace(":","-")}.log'
        cmd=[sys.executable,'-B','-u','-m',f'visual_evidence_probe.experiments.paired_depth_diagnostic.{name}']
        if indices is not None:cmd+=['--gpu',a.gpu,'--indices',','.join(map(str,indices)),'--seeds',','.join(map(str,seeds))]
        row=dict(stage=name,indices=indices,seeds=seeds,started=now(),log=str(log),blocking=blocking)
        state.update(stage=name,active_log=str(log),updated=now());state['attempts'].append(row);write(receipt,state)
        with log.open('a') as stream:result=subprocess.run(cmd,env=dict(os.environ,CUDA_LAUNCH_BLOCKING='1' if blocking else '0',CUDA_VISIBLE_DEVICES=a.gpu),stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,finished=now());state['updated']=now();write(receipt,state)
        if result.returncode:
            content=log.read_text();error=RuntimeError(f'{name} failed: {log}')
            # The regular FlexiCubes volume index was one element short in
            # the asynchronous repeat audit. Retry this specific runtime
            # signature in a fresh synchronous process, not arbitrary errors.
            error.cuda_error=('CUDA error' in content or ('flexicube.py' in content and 'vol_idx[surf_cubes]' in content and 'The shape of the mask' in content))
            raise error
    def batch(indices,seeds):
        pending=[i for i in indices if any(not (folder(i,s)/'optimization_complete.json').exists() for s in seeds)]
        if pending:
            if a.evaluation_only:raise ValueError('all4 trajectories must seal before evaluation')
            try:stage('run',pending,seeds)
            except RuntimeError as error:
                if not getattr(error,'cuda_error',False):raise
                stage('run',pending,seeds,blocking=True)
        pending=[i for i in indices if any(not (folder(i,s)/'complete.json').exists() for s in seeds)]
        if pending:stage('evaluate',pending,seeds)
        stage('report');stage('figures',indices,seeds)
    try:
        if a.closeout_only:
            if any(not (folder(i,s)/'complete.json').exists() for i in PARAMS['indices'] for s in PARAMS['seeds']):raise ValueError('primary experiments incomplete')
            stage('closeout',blocking=a.blocking)
            state.update(status='complete',stage='complete',updated=now());write(receipt,state);return
        if a.evaluation_only:
            batch(list(map(int,a.indices.split(','))),list(map(int,a.seeds.split(','))))
            state.update(status='subset_complete',stage='complete',updated=now());write(receipt,state);return
        batch([5],[42]);state['smoke_complete']=True;write(receipt,state)
        batch(PARAMS['indices'],PARAMS['seeds'])
        try:stage('closeout',blocking=a.blocking)
        except RuntimeError as error:
            if a.blocking or not getattr(error,'cuda_error',False):raise
            stage('closeout',blocking=True)
        state.update(status='complete',stage='complete',updated=now());write(receipt,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(receipt,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
