import fcntl
import os
import subprocess
import time
import traceback

from .config import OUT,ROOT,BASE,PARAMS,ARMS,read,write,now,setup


def main():
    setup();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'pipeline.json'
    if path.exists():write(OUT/'pipeline_history'/f'{now().replace(":","-")}.json',read(path))
    state=dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES','6'),status='running',updated=now(),attempts=[])
    write(path,state)
    def stage(action,args=(),label=None):
        name=label or action;log=logs/f'{name}_{now().replace(":","-")}.log'
        state.update(stage=name,active_log=str(log),updated=now(),status='running')
        state['attempts'].append(dict(stage=name,log=str(log),started=now()));write(path,state)
        with log.open('a') as stream:
            child=subprocess.Popen(['bash',str(ROOT/'run_diagnostic.sh'),action,*args],
                  cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,T2_DIAGNOSTIC_GPU=state['gpu']))
            state['child_pid']=child.pid;write(path,state)
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:state['updated']=now();write(path,state)
        state['attempts'][-1].update(returncode=code,finished=now());write(path,state)
        if code:raise RuntimeError(f'{name} failed: {log}')
    def headroom():
        while True:
            output=subprocess.check_output(['nvidia-smi','--id='+state['gpu'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(output.strip())>=14500:return
            state.update(status='waiting_gpu_memory',updated=now());write(path,state);time.sleep(30)
    try:
        if not read(BASE/'closeout.json')['complete']:raise RuntimeError('original native pilot must be complete')
        stage('prepare')
        for index in PARAMS['indices']:
            for seed in PARAMS['seeds']:
                for arm in ARMS:
                    if (OUT/'objects'/f'{index:03d}'/str(seed)/arm/'complete.json').exists():continue
                    headroom();stage('runner',('--index',str(index),'--seed',str(seed),'--arm',arm),f'{index:03d}_{seed}_{arm}')
                if not (OUT/'objects'/f'{index:03d}'/str(seed)/'result.json').exists():
                    headroom();stage('evaluate',('--index',str(index),'--seed',str(seed)),f'{index:03d}_{seed}_evaluate')
                stage('report')
        from .report import build,verify
        value=build();checks=verify()
        if not value['complete'] or not read(OUT/'figures/manifest.json')['complete']:raise RuntimeError('incomplete diagnostic')
        marker='### 4.2 大误差梯度与纠错能力诊断完整结果'
        if marker not in (ROOT/'审计.md').read_text():
            with (ROOT/'审计.md').open('a') as stream:stream.write('\n\n'+marker+'\n\n'+(OUT/'REPORT.md').read_text())
        write(OUT/'closeout.json',dict(time=now(),complete=True,cases=8,groups=32,checks=checks,report=str(OUT/'REPORT.md')))
        state.update(status='complete',stage='complete',updated=now());write(path,state)
    except BaseException as e:
        state.update(status='failed',error=str(e),traceback=traceback.format_exc(),updated=now());write(path,state);raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
