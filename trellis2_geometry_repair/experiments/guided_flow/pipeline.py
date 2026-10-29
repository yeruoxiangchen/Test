import argparse
import fcntl
import os
import subprocess
import time
import traceback

from .config import ROOT,OUT,PARAMS,case,read,write,now,setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');p.add_argument('--aux',action='store_true');a=p.parse_args()
    indices=list(map(int,a.indices.split(',')))
    if not set(indices)<=set(PARAMS['indices']):raise ValueError('outside frozen development cohort')
    setup();gpu=os.environ['CUDA_VISIBLE_DEVICES'];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    workers=OUT/'workers';workers.mkdir(exist_ok=True)
    handle=(workers/f'gpu{gpu}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=workers/f'gpu{gpu}.json'
    if path.exists():write(OUT/'pipeline_history'/f'gpu{gpu}_{now().replace(":","-")}.json',read(path))
    state=dict(gpu=gpu,aux=a.aux,indices=indices,status='running',updated=now(),attempts=[]);write(path,state)
    def stage(action,args=(),label=None):
        label=label or action;log=logs/f'gpu{gpu}_{label}_{now().replace(":","-")}.log'
        state.update(stage=label,active_log=str(log),status='running',updated=now());state['attempts'].append(dict(stage=label,log=str(log),started=now()));write(path,state)
        with log.open('a') as stream:
            child=subprocess.Popen(['bash',str(ROOT/'run_guided_flow.sh'),action,*args],cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,T2_GUIDED_GPU=gpu))
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:state['updated']=now();write(path,state)
        state['attempts'][-1].update(finished=now(),returncode=code);write(path,state)
        if code:raise RuntimeError(f'{label} failed: {log}')
    def memory(receipt=None):
        while int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<18000:
            if receipt is not None and receipt.exists():return
            state.update(status='waiting_gpu_memory',updated=now());write(path,state);time.sleep(30)
    try:
        if not a.aux:
            stage('prepare')
            if not (OUT/'validation/smoke_complete.json').exists():memory();stage('smoke')
        elif not (OUT/'validation/smoke_complete.json').exists():raise RuntimeError('primary smoke must pass first')
        if a.aux:
            while not (case(5,42)/'result.json').exists():
                state.update(status='waiting_first_case',stage='first_case_validation',updated=now());write(path,state);time.sleep(20)
        for i in indices:
            for s in PARAMS['seeds']:
                root=case(i,s);args=('--index',str(i),'--seed',str(s))
                stages=[('generation','generation_method.json',('--mode','method')),('selection','method_complete.json',('--mode','method')),
                    ('generation','generation_diagnostic.json',('--mode','diagnostic')),('selection','diagnostic_complete.json',('--mode','diagnostic')),('evaluate','result.json',())]
                for action,receipt,extra in stages:
                    dest=root/receipt
                    if dest.exists():continue
                    memory(dest)
                    if dest.exists():continue
                    stage(action,args+extra,f'{i:03d}_{s}_{action}_{extra[-1] if extra else "run"}')
                if not a.aux:stage('report')
        if not a.aux:
            from .report import build,verify
            result=build()
            if not result['complete']:raise RuntimeError('primary cohort is incomplete')
            state.update(stage='artifact_validation',updated=now());write(path,state)
            checks=verify()
            marker='### 6.2 V3完整实验结果'
            if marker not in (ROOT/'审计.md').read_text():
                with (ROOT/'审计.md').open('a') as stream:stream.write('\n\n'+marker+'\n\n'+(OUT/'REPORT.md').read_text().rstrip()+'\n')
            write(OUT/'closeout.json',dict(time=now(),complete=True,cases=8,checks=checks))
        state.update(status='subset_complete' if a.aux else 'complete',stage='complete',updated=now());write(path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(path,state);raise
    finally:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

if __name__=='__main__':main()
