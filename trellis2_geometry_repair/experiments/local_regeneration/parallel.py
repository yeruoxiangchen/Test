"""Optional spare-GPU subset; the primary pipeline owns global closeout."""
import argparse
import fcntl
import os
import subprocess
import time
import traceback

from .config import OUT,ROOT,PARAMS,case,read,write,now,setup


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--indices',default='10,11');args=parser.parse_args()
    setup();indices=list(map(int,args.indices.split(',')))
    if not indices or not set(indices)<=set(PARAMS['indices']):raise ValueError('outside frozen development cohort')
    if not (OUT/'validation/smoke_complete.json').exists():raise RuntimeError('primary GPU validation must precede subset workers')
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];lock=(OUT/f'parallel_gpu{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/f'parallel_gpu{gpu}.json';logs=OUT/'logs';logs.mkdir(exist_ok=True)
    if path.exists():write(OUT/'pipeline_history'/f'parallel_gpu{gpu}_{now().replace(":","-")}.json',read(path))
    state=dict(gpu=gpu,indices=indices,status='running',updated=now());write(path,state)
    def stage(action,index,seed,extra):
        label=f'gpu{gpu}_{index:03d}_{seed}_{action}_{extra[-1] if extra else "run"}'
        log=logs/f'{label}_{now().replace(":","-")}.log'
        state.update(status='running',stage=label,active_log=str(log),updated=now());write(path,state)
        with log.open('a') as stream:
            child=subprocess.Popen(['bash',str(ROOT/'run_local_regeneration.sh'),action,
                '--index',str(index),'--seed',str(seed),*extra],cwd=ROOT.parent,
                stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,T2_LOCAL_GPU=gpu))
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:state['updated']=now();write(path,state)
        if code:raise RuntimeError(f'{label} failed: {log}')
    try:
        for index in indices:
            for seed in PARAMS['seeds']:
                for action,receipt,extra in [('planning','plan.json',()),('generation','ss_complete.json',('--stage','ss')),
                    ('generation','sampling_complete.json',('--stage','shape')),('selection','input_complete.json',()),('evaluate','result.json',())]:
                    if (case(index,seed)/receipt).exists():continue
                    while int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<17000:
                        if (case(index,seed)/receipt).exists():break
                        state.update(status='waiting_gpu_memory',updated=now());write(path,state);time.sleep(30)
                    if (case(index,seed)/receipt).exists():continue
                    stage(action,index,seed,extra)
        state.update(status='subset_complete',stage='complete',updated=now());write(path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(path,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
