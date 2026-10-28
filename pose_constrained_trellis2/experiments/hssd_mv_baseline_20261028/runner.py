"""Bounded single-GPU queue with receipts, safe stop, and automatic final review files."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from . import common as c

DEFAULT_GPUS='0,1,2,3,4,5,6,7'

def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(') ',1)[1].split()[19]
    except (OSError,ValueError):return None


def training(stage,smoke=False):
    folder=c.OUT/('smoke/training' if smoke else 'training')/stage
    limit=2 if smoke else c.CONFIG['steps']
    args=[c.PYTHON,'-B','-m',c.native.MODULE+'.invoke','train','--stage',stage,
        '--manifest',str(c.DATA/'train.json'),'--output-dir',str(folder),
        '--model-path',c.models()[stage+'_flow'],'--max-steps',str(limit),
        '--warmup-steps',str(c.CONFIG['warmup']),'--save-every',str(c.CONFIG['save_every']),
        '--log-every','10' if not smoke else '1','--p-uncond',str(c.CONFIG['p_uncond']),
        '--min-condition-views','2','--max-condition-views','4','--context-mode','first',
        '--seed','42','--verify-cache-hashes']
    if smoke:args+=['--indices',str(c.plan()['smoke_train_index'])]
    files=sorted((folder/'checkpoints').glob('step_*.pt'))
    if files:args+=['--resume',str(files[-1])]
    return args,[folder/'checkpoints'/f'step_{limit:06d}.pt']


def stages(mode):
    p=c.plan();jobs=[]
    for stage in ('ss','shape'):
        args,outputs=training(stage,True);jobs.append(('smoke_'+stage,args,outputs,True))
    for arm in c.CONFIG['arms']:
        args=[c.PYTHON,'-B','-m',c.MODULE+'.evaluate','--arm',arm,'--uid',p['smoke_eval_uid'],'--seed','42','--smoke']
        outputs=[c.OUT/'smoke/evaluation'/arm/p['smoke_eval_uid']/'seed_42/result.json']
        jobs.append(('smoke_eval_'+arm,args,outputs,True))
    jobs.append(('qualify_pairs',[c.PYTHON,'-B','-m',c.MODULE+'.qualify_pairs'],[c.DATA/'pair_qualification_gt_only.json'],True))
    if mode=='preflight':return jobs
    for stage in ('ss','shape'):
        args,outputs=training(stage);jobs.append(('train_'+stage,args,outputs,True))
    for obj in p['objects']:
        for seed in c.CONFIG['evaluation_seeds']:
            for arm in c.CONFIG['arms']:
                uid=obj['uid'];args=[c.PYTHON,'-B','-m',c.MODULE+'.evaluate','--arm',arm,'--uid',uid,'--seed',str(seed)]
                jobs.append((f'eval_{uid}_{seed}_{arm}',args,[c.OUT/'evaluations'/arm/uid/f'seed_{seed}/result.json'],True))
    jobs.append(('summary_and_figures',[c.PYTHON,'-B','-m',c.MODULE+'.summarize'],[c.OUT/'summary.json',c.OUT/'REPORT.md'],True))
    return jobs


def choose_gpu():
    allowed=os.environ.get('T2_BASE_GPUS',DEFAULT_GPUS).split(',')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    occupied=set()
    for line in proc.splitlines():
        if ',' not in line:continue
        uuid,memory=[v.strip() for v in line.split(',')]
        if not memory.isdigit() or int(memory)>=512:occupied.add(uuid)
    for line in output.splitlines():
        index,uuid,used,free,util=[x.strip() for x in line.split(',')]
        if index in allowed and uuid not in occupied and int(used)<512 and int(free)>=22000 and int(util)==0:return int(index)
    return None


def run(mode):
    env=c.environment();os.environ.update(env);c.OUT.mkdir(parents=True,exist_ok=True)
    with (c.OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=c.source_identity();c.immutable(c.OUT/'runtime.json',key)
        jobs=stages(mode);child=None;active=None;gpu=None;log=None
        base={'pid':os.getpid(),'process_start':process_start(os.getpid()),'runtime_identity':key['identity'],'mode':mode,'total_stages':len(jobs)}
        def status(state,**kw):c.write(c.OUT/'status.json',{**base,'time':c.now(),'status':state,'stage':active,'gpu':gpu,'log':str(log) if log else None,**kw})
        def stop(signum,frame):raise KeyboardInterrupt('Stop requested')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            for index,(name,args,outputs,needs_gpu) in enumerate(jobs):
                active=name;gpu=None
                receipt=c.OUT/'stages'/f'{name}.json'
                if receipt.exists():
                    old=c.read(receipt)
                    assert old['runtime_identity']==key['identity'],'Stage source binding changed'
                    for file in old['artifacts']:
                        p=Path(file['path']);assert p.is_file() and p.stat().st_size==file['bytes'],p
                    continue
                while needs_gpu and gpu is None:
                    gpu=choose_gpu()
                    if gpu is None:status('waiting_for_gpu',completed_stages=index,candidates=os.environ.get('T2_BASE_GPUS',DEFAULT_GPUS));time.sleep(10)
                # Refresh resume path only at execution time, not early queue construction.
                if name.startswith(('train_','smoke_')) and name.split('_')[-1] in ('ss','shape'):
                    args,outputs=training(name.split('_')[-1],name.startswith('smoke_'))
                childenv={**env,'CUDA_VISIBLE_DEVICES':str(gpu) if needs_gpu else ''}
                log=c.OUT/'logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w') as stream:
                    stream.write('COMMAND '+json.dumps(args)+'\n');stream.flush()
                    child=subprocess.Popen(args,cwd=c.REPO,env=childenv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        status('running',completed_stages=index,child_pid=child.pid,child_start=process_start(child.pid));time.sleep(5)
                    if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; see {log}')
                child=None
                if name.startswith(('eval_','smoke_eval_')):
                    result=c.read(outputs[0]);folder=outputs[0].parent
                    outputs=outputs+[folder/'ss.npz']
                    if result['success']:
                        outputs += [folder/'mesh.ply',folder/'mesh_world.ply',folder/'shape.npz']
                artifacts=[]
                for p in outputs:
                    if not p.is_file() or not p.stat().st_size:raise RuntimeError('Missing artifact '+str(p))
                    artifacts.append({'path':str(p),'bytes':p.stat().st_size})
                c.write(receipt,{'time':c.now(),'runtime_identity':key['identity'],'artifacts':artifacts,'log':str(log)})
            active=None;gpu=None
            status('preflight_complete' if mode=='preflight' else 'all_complete_review_required',completed_stages=len(jobs))
        except BaseException as e:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            status('stopped' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e))
            if not isinstance(e,KeyboardInterrupt):raise


def launch(mode):
    env=c.environment();c.OUT.mkdir(parents=True,exist_ok=True)
    with (c.OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (c.OUT/'queue.lock').open('a') as q:
            try:fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('已有队列运行，使用watch。')
        # Also cover the small interval before the new runner acquires its lock.
        last=c.OUT/'launcher.json'
        if last.exists():
            prior=c.read(last)
            if process_start(prior['pid'])==prior.get('process_start'):raise SystemExit('队列已启动，等待初始化或使用watch。')
        if not (c.OUT/'plan.json').exists():raise SystemExit('先执行run.sh prepare')
        if (c.OUT/'runtime.json').exists() and c.read(c.OUT/'runtime.json')!=c.source_identity():raise SystemExit('源码身份变更，拒绝混用旧回执。')
        with (c.OUT/'launcher.log').open('a') as log:
            p=subprocess.Popen([c.PYTHON,'-B','-m',c.MODULE+'.runner','run','--mode',mode],cwd=c.REPO,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        c.write(last,{'pid':p.pid,'process_start':process_start(p.pid),'time':c.now(),'mode':mode})
        print(f'已启动 {mode}，PID {p.pid}，候选GPU {env.get("T2_BASE_GPUS",DEFAULT_GPUS)}；单卡串行，仅空卡进入。')


def show():
    if not (c.OUT/'status.json').exists():print('尚未启动');return
    s=c.read(c.OUT/'status.json')
    alive=process_start(s['pid'])==s.get('process_start')
    print(f'[{c.now()}] HSSD TRELLIS.2 MVBase12\n状态：{s["status"]}  阶段：{s.get("stage") or "无活动阶段"}  GPU：{s.get("gpu")}')
    print(f'阶段：{s.get("completed_stages",0)}/{s["total_stages"]}；PID：{s["pid"]}；进程身份：{"匹配" if alive else "已结束"}；更新：{s["time"]}')
    if s['status']=='waiting_for_gpu':print('等待候选卡：'+s.get('candidates',DEFAULT_GPUS)+'；显存<512MiB、空余>=22000MiB、利用率0且无大显存compute进程。')
    for stage in ('ss','shape'):
        files=sorted((c.OUT/'training'/stage/'checkpoints').glob('step_*.pt'))
        step=int(files[-1].stem.split('_')[-1]) if files else 0
        logs=sorted((c.OUT/'logs').glob(f'train_{stage}_*.log'))
        last=None
        if logs:
            with logs[-1].open('rb') as f:f.seek(max(0,logs[-1].stat().st_size-8000));tail=f.read().decode(errors='replace')
            for line in tail.splitlines():
                if line.startswith('{'):
                    try:last=json.loads(line)
                    except json.JSONDecodeError:pass
        print(f'  {stage}: {max(step,last.get("step",0) if last else 0)}/400'+(f' loss={last.get("loss"):.6f}' if last and isinstance(last.get('loss'),(float,int)) else ''))
    for arm in c.CONFIG['arms']:
        rows=[c.read(p) for p in (c.OUT/'evaluations'/arm).glob('*/seed_*/result.json')]
        print(f'  {arm}: 完成{len(rows)}/24；有效mesh={sum(r["success"] for r in rows)}')
    p=c.DATA/'pair_qualification_gt_only.json'
    if p.exists():
        d=c.read(p);print(f'邻居组：{len(d["groups"])}；确认遮挡：{d["verified_occlusion_pairs"]}；两端已有原生标签：{d["native_training_ready_pairs"]}')
    if s.get('log'):print('日志：'+s['log'])
    if s.get('error'):print('错误：'+s['error'])
    if s['status']=='all_complete_review_required':print('训练、48次生成及网格图汇总已结束，无需继续等待。')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','start','status','watch','stop','refreeze-empty']);p.add_argument('--mode',choices=['preflight','all'],default='all');a=p.parse_args()
    if a.action=='run':run(a.mode)
    elif a.action=='start':launch(a.mode)
    elif a.action=='refreeze-empty':
        with (c.OUT/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if list((c.OUT/'stages').glob('*.json')):raise RuntimeError('Completed stages exist; cannot refreeze')
            if list((c.OUT/'logs').glob('*.log')):raise RuntimeError('Actual stage execution already began; review before changing identity')
            folder=c.OUT/'repairs'/str(time.time_ns());folder.mkdir(parents=True)
            for name in ('runtime.json','status.json','launcher.json'):
                path=c.OUT/name
                if path.exists():path.rename(folder/name)
            c.write(folder/'reason.json',{'time':c.now(),'reason':'No computation executed: allow all idle candidate GPUs and tolerate only <512MiB idle resident contexts; connect preflight to full queue'})
            print('空队列身份已归档；未删除任何计算结果。')
    elif a.action=='stop':
        s=c.read(c.OUT/'status.json')
        if process_start(s['pid'])==s.get('process_start'):os.kill(s['pid'],signal.SIGTERM);print('已请求停止自有队列及其子任务，保留检查点。')
        else:print('队列已结束。')
    elif a.action=='status':show()
    else:
        while True:show();time.sleep(10)


if __name__=='__main__':main()
