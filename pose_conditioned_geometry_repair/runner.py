"""Resource-aware background queue. A failed overfit gate blocks formal training."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from . import common as c

UNIT='pose-conditioned-geometry-repair-round1-20261029'
MODULE='pose_conditioned_geometry_repair'


def alive(pid,start=None):
    try:
        value=Path(f'/proc/{int(pid)}/stat').read_text().split()[21]
        return start is None or value==str(start)
    except (OSError,ValueError):
        return False


def gpu_inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','uuid','free','util'],[int(x[0]),x[1],int(x[2]),int(x[3])]))
            for line in out.splitlines() if (x:=[s.strip() for s in line.split(',')])]


@contextmanager
def free_gpu(label):
    locks=c.ROOT/'locks'
    locks.mkdir(parents=True,exist_ok=True)
    allowed=[int(x) for x in os.environ.get('T2_REPAIR_GPUS','3,2,5,6,7').split(',')]
    while True:
        inventory={x['index']:x for x in gpu_inventory()}
        for index in allowed:
            gpu=inventory[index]
            if gpu['free']<14500 or gpu['util']>15:
                continue
            f=(locks/f'gpu_{index}.lock').open('a')
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            # Check again after acquiring this project's lock. Other projects
            # need not use our locks; never stop or relocate their processes.
            current={x['index']:x for x in gpu_inventory()}[index]
            if current['free']<14500 or current['util']>15:
                f.close()
                continue
            try:
                c.write(c.OUT/'workers'/f'{label}.json',{'time':c.now(),'state':'running',
                    'gpu':index,'gpu_uuid':gpu['uuid']})
                yield gpu
            finally:
                fcntl.flock(f,fcntl.LOCK_UN)
                f.close()
            return
        c.write(c.OUT/'workers'/f'{label}.json',{'time':c.now(),'state':'waiting_for_free_gpu',
            'allowed':allowed,'required_free_mib':14500})
        time.sleep(20)


def command(label,module,args,gpu=False,required=None):
    receipt=c.OUT/'receipts'/f'{label}.json'
    if receipt.exists() and c.read(receipt).get('complete') and all(Path(p).exists() for p in (required or [])):
        return
    log=c.OUT/'logs'/f'{label}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[c.PYTHON,'-B','-m',MODULE+'.'+module,*map(str,args)]
    def execute(assignment=None):
        env=c.environment()
        if assignment:
            env['CUDA_VISIBLE_DEVICES']=assignment['uuid']
        with log.open('a',buffering=1) as f:
            f.write('\n'+c.now()+' COMMAND '+json.dumps(cmd)+'\n')
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=c.REPO)
            c.write(c.OUT/'workers'/f'{label}.json',{'time':c.now(),'state':'running',
                'pid':proc.pid,'process_start':Path(f'/proc/{proc.pid}/stat').read_text().split()[21],
                'log':str(log),'gpu':assignment['index'] if assignment else None})
            rc=proc.wait()
        if rc:
            c.write(c.OUT/'workers'/f'{label}.json',{'time':c.now(),'state':'failed','returncode':rc,'log':str(log)})
            raise RuntimeError(label+' failed; see '+str(log))
        if required and not all(Path(p).exists() for p in required):
            raise RuntimeError(label+' returned success without required artifacts')
        value={'time':c.now(),'complete':True,'log':str(log),'command':cmd,
               'gpu':assignment,'required':list(map(str,required or []))}
        c.write(receipt,value)
        c.write(c.OUT/'workers'/f'{label}.json',{'time':c.now(),'state':'complete','log':str(log)})
    if gpu:
        with free_gpu(label) as assignment:
            execute(assignment)
    else:
        execute()


def training(stage,arm,subset,steps,initial=0):
    name=f'{subset}_{arm}_{stage}'
    out=c.OUT/name
    args=['--stage',stage,'--arm',arm,'--subset',subset,'--output',out,
          '--steps',steps,'--save-every',200,'--log-every',10,
          '--warmup',30 if subset=='overfit' else 100,'--grad-accum',1 if subset=='overfit' else 4]
    last=out/'checkpoints/last.pt'
    if last.exists():
        args+=['--resume',last]
    command(name+f'_step{steps}','train',args,gpu=True,
            required=[out/'checkpoints'/f'step_{steps:06d}.pt',out/'complete.json'])
    return out/'checkpoints'/f'step_{steps:06d}.pt'


def evaluation(stage,arm,subset,checkpoint,ss_checkpoint='',limit=32,seed=42,variants=(0,)):
    steps=Path(checkpoint).stem
    name=f'eval_{subset}_{arm}_{stage}_{steps}_s{seed}'
    out=c.OUT/name
    args=['--stage',stage,'--arm',arm,'--subset',subset,'--checkpoint',checkpoint,
          '--output',out,'--limit',limit,'--seed',seed,'--variants',*variants]
    if ss_checkpoint:
        args+=['--ss-checkpoint',ss_checkpoint]
    command(name,'evaluate',args,gpu=True,required=[out/'summary.json'])
    result=c.read(out/'summary.json')
    if not result['complete']:
        raise RuntimeError('Evaluation returned incomplete results')
    return result


def overfit_stage(stage,ss_checkpoint=''):
    first=600 if stage=='ss' else 1000
    for steps in [first,2000]:
        ck=training(stage,'repair','overfit',steps)
        summary=evaluation(stage,'repair','overfit',ck,ss_checkpoint)
        if summary['overfit_gate_passed']:
            return str(ck),summary
    raise RuntimeError(stage+' overfit gate failed at 2000 updates; formal training is blocked')


def formal_job(arm,stage):
    return str(training(stage,arm,'train',4000))


def pipeline():
    c.OUT.mkdir(parents=True,exist_ok=True)
    lock=(c.OUT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.getpid()
    started=Path(f'/proc/{pid}/stat').read_text().split()[21]
    def phase(name,**kwargs):
        c.write(c.OUT/'status.json',{'time':c.now(),'state':'running','phase':name,
            'pid':pid,'process_start':started,**kwargs})
    try:
        phase('preflight')
        command('freeze','prepare',['freeze'],required=[c.DATA/'plan.json'])
        # Archive a small source snapshot on /data; nothing touches another
        # project's source, model checkpoints, active services or git index.
        snap=c.OUT/'source_snapshot'
        for p in c.HERE.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.md','.json'] and '__pycache__' not in p.parts:
                dst=snap/p.relative_to(c.HERE)
                dst.parent.mkdir(parents=True,exist_ok=True)
                dst.write_bytes(p.read_bytes())
        c.write(c.OUT/'source_frozen.json',c.source_identity())
        # Tests use the base Python where pytest is installed; no CUDA needed.
        testlog=c.OUT/'logs/tests.log'
        testlog.parent.mkdir(parents=True,exist_ok=True)
        with testlog.open('w') as f:
            subprocess.run(['bash',str(c.HERE/'run.sh'),'tests'],stdout=f,stderr=subprocess.STDOUT,check=True,env=c.environment())
        phase('prepare_overfit32')
        command('prepare_overfit32','prepare',['objects','--subset','overfit'],gpu=True)
        command('publish_overfit32','prepare',['publish','--subset','overfit'],required=[c.DATA/'overfit.json'])
        phase('training_smoke_and_resume')
        for stage in ['ss','shape']:
            out=c.OUT/f'smoke_{stage}_queued'
            args=['--stage',stage,'--subset','overfit','--output',out,'--steps',3,
                  '--save-every',3,'--log-every',1,'--warmup',1]
            command('smoke_'+stage,'train',args,gpu=True,required=[out/'checkpoints/step_000003.pt'])
            args[args.index('--steps')+1]=5
            args+=['--resume',out/'checkpoints/step_000003.pt']
            command('resume_smoke_'+stage,'train',args,gpu=True,required=[out/'checkpoints/step_000005.pt'])
        phase('native_inference_and_mesh_smoke')
        command('evaluation_smoke','evaluate',['--stage','shape','--subset','overfit',
            '--checkpoint',c.OUT/'smoke_shape_queued/checkpoints/step_000005.pt',
            '--ss-checkpoint',c.OUT/'smoke_ss_queued/checkpoints/step_000005.pt',
            '--output',c.OUT/'evaluation_smoke','--limit',1],gpu=True,
            required=[c.OUT/'evaluation_smoke/summary.json'])
        # Validate the full multiview control's actual CUDA train/sampler path
        # before allocating the four formal training jobs.
        for stage in ['ss','shape']:
            output=c.OUT/f'smoke_full_{stage}'
            command('smoke_full_'+stage,'train',['--stage',stage,'--arm','full',
                '--subset','overfit','--output',output,'--steps',3,'--save-every',3,
                '--log-every',1,'--warmup',1],gpu=True,
                required=[output/'checkpoints/step_000003.pt'])
        command('evaluation_full_smoke','evaluate',['--stage','shape','--arm','full',
            '--subset','overfit','--checkpoint',c.OUT/'smoke_full_shape/checkpoints/step_000003.pt',
            '--ss-checkpoint',c.OUT/'smoke_full_ss/checkpoints/step_000003.pt',
            '--output',c.OUT/'evaluation_full_smoke','--limit',1],gpu=True,
            required=[c.OUT/'evaluation_full_smoke/summary.json'])
        phase('overfit32_ss')
        ss,ss_summary=overfit_stage('ss')
        phase('overfit32_shape')
        shape,shape_summary=overfit_stage('shape',ss)
        c.write(c.OUT/'overfit_gate.json',{'time':c.now(),'passed':True,
            'ss_checkpoint':ss,'shape_checkpoint':shape,'ss':ss_summary,'shape':shape_summary,
            'scope':'32 training objects, fixed missing ROI; not held-out generalization'})
        phase('prepare_formal2000_dev64')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(command,'prepare_formal_rank'+str(r),'prepare',
                ['objects','--subset','all','--rank',r,'--world',2],True) for r in range(2)]
            for f in futures:
                f.result()
        for subset in ['train','dev']:
            command('publish_'+subset,'prepare',['publish','--subset',subset],required=[c.DATA/f'{subset}.json'])
        phase('formal_training')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={(arm,stage):pool.submit(formal_job,arm,stage)
                     for arm in ['repair','full'] for stage in ['ss','shape']}
            checkpoints={k:f.result() for k,f in futures.items()}
        c.write(c.OUT/'formal_checkpoints.json',{'time':c.now(),
            'checkpoints':{arm:{stage:checkpoints[arm,stage] for stage in ['ss','shape']} for arm in ['repair','full']}})
        phase('formal_dev64_evaluation')
        summaries={}
        for arm in ['repair','full']:
            for seed in [42,43]:
                result=evaluation('shape',arm,'dev',checkpoints[arm,'shape'],
                    checkpoints[arm,'ss'],limit=64,seed=seed,variants=(0,1,2,3))
                summaries[f'{arm}_{seed}']=result
        c.write(c.OUT/'formal_summary.json',{'time':c.now(),'complete':True,'summaries':summaries,
            'interpretation':'synthetic corruptions, known ROI; raw generated errors and automatic localization remain separate tests'})
        phase('report')
        command('report','report',[],required=[c.OUT/'REPORT.md'])
        c.write(c.OUT/'status.json',{'time':c.now(),'state':'complete','phase':'complete','pid':pid,'process_start':started})
    except Exception as e:
        c.write(c.OUT/'status.json',{'time':c.now(),'state':'failed','phase':'failed','error':str(e),
            'pid':pid,'process_start':started})
        raise
    finally:
        lock.close()


def launch():
    state=c.OUT/'status.json'
    if state.exists():
        s=c.read(state)
        if s['state']=='running' and alive(s['pid'],s['process_start']):
            print('Queue already running',s)
            return
    env=c.environment()
    args=['systemd-run','--user','--unit='+UNIT,'--collect','--property=Nice=10',
          '--property=IOSchedulingClass=best-effort','--property=IOSchedulingPriority=7',
          '--property=WorkingDirectory='+str(c.REPO),
          '--setenv=T2_REPAIR_GPUS='+os.environ.get('T2_REPAIR_GPUS','3,2,5,6,7'),
          '/bin/bash',str(c.HERE/'run.sh'),'pipeline']
    subprocess.run(args,env=env,check=True)
    print('Started '+UNIT)


def status():
    path=c.OUT/'status.json'
    if path.exists():
        s=c.read(path)
        print('队列：',s['state'],'；阶段：',s['phase'],'；更新：',s['time'])
        if s['state']=='running' and not alive(s['pid'],s['process_start']):
            print('警告：记录中的队列进程已不在运行')
        if s.get('error'):
            print('原因：',s['error'])
    else:
        print('队列尚未启动')
    if (c.DATA/'plan.json').exists():
        rows=c.sample_rows()
        done={r['object_uid'] for r in rows if (c.DATA/'objects'/r['object_uid']/'complete.json').exists()}
        rejected={r['object_uid'] for r in rows if (c.DATA/'objects'/r['object_uid']/'rejected.json').exists()}
        if (c.DATA/'formal_plan.json').exists():
            selected=c.read(c.DATA/'formal_plan.json')['plan']['splits']
            selected_uids=set(selected['train'])|set(selected['dev'])
            print(f'正式原生缓存：{len(done & selected_uids)}/2064；训练2000；Dev64')
        else:
            print(f'候选校验：{len(done|rejected)}/{len(rows)}；有效 {len(done)}；排除 {len(rejected)}；目标2000+64')
        print(f'32对象：{sum(r["overfit32"] and r["object_uid"] in done for r in rows)}/32')
    gate=c.OUT/'overfit_gate.json'
    print('过拟合检查：','通过' if gate.exists() and c.read(gate)['passed'] else '尚未通过或尚未完成')
    for name in ['overfit_repair_ss','overfit_repair_shape','train_repair_ss','train_repair_shape','train_full_ss','train_full_shape']:
        p=c.OUT/name/'progress.json'
        if p.exists():
            r=c.read(p)
            config=c.OUT/name/'config.json'
            target=c.read(config)['args']['steps'] if config.exists() else None
            fraction=f'{r["step"]}/{target}' if target else str(r['step'])
            print(f'{name}：{fraction} updates；loss={r["flow_loss"]:.6f}；更新 {r["time"]}')
    progress_files=sorted([*c.OUT.glob('eval*/progress.json'),*c.OUT.glob('dev64*/progress.json')])
    for p in progress_files:
        r=c.read(p)
        if 'records' in r:
            print(f'{p.parent.name}：{r.get("stage", "评估")} {r["records"]}/{r["expected"]}；更新 {r["time"]}')
    for p in sorted((c.OUT/'workers').glob('*.json')) if (c.OUT/'workers').exists() else []:
        r=c.read(p)
        if r['state']!='complete':
            print(p.stem,r['state'],'GPU',r.get('gpu'),'日志',r.get('log',''))
    print('结果目录：',c.OUT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['launch','pipeline','status'])
    a=p.parse_args()
    {'launch':launch,'pipeline':pipeline,'status':status}[a.action]()


if __name__=='__main__':
    main()
