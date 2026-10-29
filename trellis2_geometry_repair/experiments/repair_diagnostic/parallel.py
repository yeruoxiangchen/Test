"""Optional spare-GPU subset; the original manager owns global closeout."""
import argparse
import os
import subprocess
import time
import traceback

from .config import OUT,ROOT,PARAMS,ARMS,case,read,write,now,setup


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='10,11');a=p.parse_args();setup()
    indices=list(map(int,a.indices.split(',')))
    if not set(indices)<=set(PARAMS['indices']):raise RuntimeError('outside frozen cohort')
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    path=OUT/f'parallel_gpu{gpu}.json';state=dict(gpu=gpu,status='running',updated=now(),indices=indices)
    write(path,state)
    def stage(action,index,seed,arm=None):
        label=f'gpu{gpu}_{index:03d}_{seed}_{arm or action}';log=logs/f'{label}_{now().replace(":","-")}.log'
        args=['--index',str(index),'--seed',str(seed)]+(['--arm',arm] if arm else [])
        state.update(stage=label,active_log=str(log),updated=now(),status='running');write(path,state)
        with log.open('a') as stream:
            child=subprocess.Popen(['bash',str(ROOT/'run_diagnostic.sh'),action,*args],cwd=ROOT.parent,
                  stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,T2_DIAGNOSTIC_GPU=gpu))
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:state['updated']=now();write(path,state)
        if code:raise RuntimeError(f'{label} failed: {log}')
    try:
        if not (OUT/'protocol.json').exists():raise RuntimeError('main prepare must precede parallel workers')
        for index in indices:
            for seed in PARAMS['seeds']:
                for arm in ARMS:
                    if (case(index,seed)/arm/'complete.json').exists():continue
                    while int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<14500:
                        state.update(status='waiting_gpu_memory',updated=now());write(path,state);time.sleep(30)
                    stage('runner',index,seed,arm)
                stage('evaluate',index,seed)
        state.update(status='subset_complete',stage='complete',updated=now());write(path,state)
    except BaseException as e:
        state.update(status='failed',error=str(e),traceback=traceback.format_exc(),updated=now());write(path,state);raise


if __name__=='__main__':main()
