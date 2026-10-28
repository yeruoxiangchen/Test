import fcntl
import os
import subprocess
from pathlib import Path
import traceback

from .config import OUT,ROOT,PARAMS,folder,read,write,now


def main():
    OUT.mkdir(parents=True,exist_ok=True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=OUT/'pipeline.json'
    if state_path.exists():write(OUT/'pipeline_history'/f'{now().replace(":","-")}.json',read(state_path))
    state=dict(time=now(),updated=now(),status='running',gpu=os.environ.get('CUDA_VISIBLE_DEVICES','6'),attempts=[])
    write(state_path,state)
    def stage(name,extra=()):
        log=logs/f'{name}_{now().replace(":","-")}.log'
        state.update(stage=name,active_log=str(log),updated=now());state['attempts'].append(dict(stage=name,log=str(log),started=now()));write(state_path,state)
        with log.open('a') as stream:
            process=subprocess.Popen(['bash',str(ROOT/'run.sh'),name,*extra],cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,T2_REPAIR_GPU=state['gpu']))
            state['child_pid']=process.pid;write(state_path,state)
            while True:
                try:code=process.wait(timeout=30);break
                except subprocess.TimeoutExpired:state['updated']=now();write(state_path,state)
        state['attempts'][-1].update(returncode=code,finished=now());write(state_path,state)
        if code:raise RuntimeError(f'{name} failed: {log}')
    try:
        stage('environment');stage('prepare');stage('compat')
        stage('generate',('--indices','5','--seeds','42'))
        if not (OUT/'validation/smoke_complete.json').exists():stage('smoke')
        # First natural object end-to-end; then the remaining known pilot objects.
        for indices in ('5','6,10,11'):
            stage('generate',('--indices',indices));stage('fit',('--indices',indices))
            stage('edit',('--indices',indices));stage('evaluate',('--indices',indices))
            stage('figures',('--indices',indices));stage('report')
        from .report import build,verify_artifacts
        value=build();checks=verify_artifacts()
        if not value['complete'] or not read(OUT/'figures/manifest.json')['complete']:raise RuntimeError('pilot incomplete')
        old=(ROOT/'审计.md').read_text();marker='## 3 原生几何纠错pilot完整结果'
        if marker not in old:
            with (ROOT/'审计.md').open('a') as stream:
                stream.write('\n\n'+marker+'\n\n'+(OUT/'REPORT.md').read_text())
        write(OUT/'closeout.json',dict(time=now(),complete=True,cases=8,artifact_checks=checks,
              smoke=read(OUT/'validation/smoke_complete.json'),report=str(OUT/'REPORT.md')))
        state.update(status='complete',stage='complete',updated=now());write(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(state_path,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
