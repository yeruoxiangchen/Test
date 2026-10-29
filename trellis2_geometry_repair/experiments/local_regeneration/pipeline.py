import fcntl
import os
import subprocess
import traceback
import time

from .config import ROOT,OUT,PARAMS,case,read,write,now,setup


def main():
    setup();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'pipeline.json'
    if path.exists():write(OUT/'pipeline_history'/f'{now().replace(":","-")}.json',read(path))
    state=dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES','6'),status='running',updated=now(),attempts=[]);write(path,state)
    def stage(action,args=(),label=None):
        name=label or action;log=logs/f'{name}_{now().replace(":","-")}.log'
        state.update(stage=name,active_log=str(log),updated=now(),status='running');state['attempts'].append(dict(stage=name,started=now(),log=str(log)));write(path,state)
        with log.open('a') as stream:
            child=subprocess.Popen(['bash',str(ROOT/'run_local_regeneration.sh'),action,*args],cwd=ROOT.parent,
                   stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,T2_LOCAL_GPU=state['gpu']))
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:state['updated']=now();write(path,state)
        state['attempts'][-1].update(finished=now(),returncode=code);write(path,state)
        if code:raise RuntimeError(f'{name} failed: {log}')
    def memory():
        while int(subprocess.check_output(['nvidia-smi','--id='+state['gpu'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<17000:
            state.update(status='waiting_gpu_memory',updated=now());write(path,state);time.sleep(30)
    try:
        stage('prepare')
        if not (OUT/'validation/smoke_complete.json').exists():memory();stage('smoke')
        for index in PARAMS['indices']:
            for seed in PARAMS['seeds']:
                root=case(index,seed);args=('--index',str(index),'--seed',str(seed))
                for action,receipt,extra in [('planning','plan.json',()),('generation','ss_complete.json',('--stage','ss')),
                       ('generation','sampling_complete.json',('--stage','shape')),('selection','input_complete.json',()),('evaluate','result.json',())]:
                    if (root/receipt).exists():continue
                    memory();stage(action,args+extra,f'{index:03d}_{seed}_{action}_{extra[-1] if extra else "run"}')
                stage('report')
        from .report import build,verify
        value=build();checks=verify()
        if not value['complete'] or not read(OUT/'figures/manifest.json')['complete']:raise RuntimeError('incomplete eight-case experiment')
        marker='### 5.2 原生局部再生成完整结果'
        if marker not in (ROOT/'审计.md').read_text():
            with (ROOT/'审计.md').open('a') as stream:stream.write('\n\n'+marker+'\n\n'+(OUT/'REPORT.md').read_text().rstrip()+'\n')
        write(OUT/'closeout.json',dict(time=now(),complete=True,cases=8,checks=checks,report=str(OUT/'REPORT.md')))
        state.update(status='complete',stage='complete',updated=now());write(path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),updated=now());write(path,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
