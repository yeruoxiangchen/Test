"""Resumable background execution; runtime failures never count as no-ops."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback

from .config import OUT,PARAMS,folder,registry,read,write,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='7');p.add_argument('--indices');p.add_argument('--blocking',action='store_true');p.add_argument('--wait-unit');p.add_argument('--evaluation-only',action='store_true');a=p.parse_args()
    indices=list(map(int,a.indices.split(','))) if a.indices else [r['index'] for r in registry()['rows']]
    lock=(OUT/f'pipeline_gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(started=now(),updated=now(),gpu=a.gpu,indices=indices,status='running',completed_stages=[],attempts=[])
    state_path=OUT/f'pipeline_gpu{a.gpu}.json'
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    if a.wait_unit:
        if not a.wait_unit.startswith('visual-evidence-dense-'):raise ValueError('only wait for this experiment units')
        state.update(stage='waiting_for_preflight',waiting_for=a.wait_unit);write(state_path,state)
        while subprocess.run(['systemctl','--user','is-active','--quiet',a.wait_unit]).returncode==0:time.sleep(15)
    def stage(name,selected,blocking=False,extra=None):
        args=[sys.executable,'-B','-u','-m',f'visual_evidence_probe.experiments.dense_evidence.{name}']
        if selected is not None:args+=['--gpu',a.gpu,'--indices',','.join(map(str,selected))]
        if extra:args+=extra
        stamp=now().replace(':','-');log=logs/f'pipeline_gpu{a.gpu}_{name}_{stamp}.log'
        attempt=dict(stage=name,indices=selected,time=now(),log=str(log),cuda_launch_blocking=blocking)
        state.update(stage=name,active_indices=selected,active_log=str(log),updated=now());state['attempts'].append(attempt);write(state_path,state)
        env=dict(os.environ);env['CUDA_LAUNCH_BLOCKING']='1' if blocking else '0'
        with log.open('a') as stream:result=subprocess.run(args,env=env,stdout=stream,stderr=subprocess.STDOUT)
        attempt.update(returncode=result.returncode,finished=now());state['updated']=now();write(state_path,state)
        if result.returncode:
            error=RuntimeError(f'{name} failed: {log}')
            error.cuda_failure=('CUDA error' in log.read_text() or 'device-side assert' in log.read_text())
            raise error
        state['completed_stages'].append(dict(stage=name,indices=selected,finished=now()));write(state_path,state)
    try:
        # One decoder process spans the cohort. On an asynchronous CUDA error,
        # preserve completed cases and retry only unfinished cases synchronously.
        pending=[i for i in indices if not all((folder(i,s)/'input_complete.json').exists() for s in PARAMS['seeds'])]
        if pending and a.evaluation_only:raise RuntimeError(f'input decisions not ready: {pending}')
        if pending:
            try:stage('run',pending,blocking=a.blocking)
            except RuntimeError as error:
                if a.blocking or not getattr(error,'cuda_failure',False):raise
                pending=[i for i in indices if not all((folder(i,s)/'input_complete.json').exists() for s in PARAMS['seeds'])]
                state['synchronous_retry_after_cuda_failure']=True;write(state_path,state)
                stage('run',pending,blocking=True)
        pending_evaluation=[i for i in indices if not all((folder(i,s)/'complete.json').exists() for s in PARAMS['seeds'])]
        if pending_evaluation:stage('evaluate',pending_evaluation)
        # These inputs/targets are frozen within this experiment directory.
        # depth_report merges records, so a completed full table is reusable.
        # In particular a full-cohort worker must not schedule the same table twice.
        quality_path=OUT/'depth_quality.json'
        quality=read(quality_path)['records'] if quality_path.exists() else []
        ready={(r['index'],r['seed']) for r in quality if r.get('metric_definition')=='clipped_fixed_missing_v2'}
        pending_quality=[i for i in indices if any((i,s) not in ready for s in PARAMS['seeds'])]
        if pending_quality:stage('depth_report',pending_quality)
        stage('report',None)
        stage('figures',indices)
        if all((folder(obj['index'],s)/'complete.json').exists() for obj in registry()['rows'] for s in PARAMS['seeds']):
            figures=read(OUT/'figures/manifest.json')
            if figures['complete']:stage('closeout',None)
        state.update(status='complete' if (OUT/'closeout.json').exists() else 'subset_complete',stage='complete',updated=now());write(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(state_path,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
