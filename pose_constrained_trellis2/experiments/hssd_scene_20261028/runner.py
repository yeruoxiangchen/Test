"""Bounded native T2 qualification queue. No automatic formal model training."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from . import common as c

def identity():
    files=list(c.HERE.glob('*.py'))+[c.HERE/'run.sh',c.HERE/'PROTOCOL.md']
    files+=list((c.REPO/'pose_constrained_trellis2').glob('*.py'))
    return {'files':{str(p.relative_to(c.REPO)):c.sha(p) for p in sorted(files)},'plan':c.plan()['identity'],'models':c.models()}

def gpu():
    allowed=os.environ.get('T2_SCENE_GPUS','7').split(',')
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in data.splitlines():
        index,used,free,util=[int(x.strip()) for x in line.split(',')]
        if str(index) in allowed and used<512 and free>=22000 and util==0:return index
    return None

def stages():
    result=[]
    for row in c.plan()['plan']['objects']:
        uid=row['uid'];out=c.DATA/'objects'/uid
        for action,res in [('dual',512),('encode',512),('dual',1024),('encode',1024),('ss',0),('decode',0),('pose',0)]:
            name=f'{uid}_{action}'+(str(res) if res else '')
            args=[c.PYTHON,'-B','-m',c.MODULE+'.worker',action,'--uid',uid]
            if res:args+=['--resolution',str(res)]
            outputs={'dual':[out/f'dual_{res}.vxz',out/'transform.json',out/'target_t2.ply'],
               'encode':[out/f'shape_{res}.npz'],'ss':[out/'ss.npz',out/'ss.json'],
               'decode':[out/'decoded_t2.ply',out/'decoded_world.ply',out/'codec.json',out/'codec.png'],
               'pose':[out/'pose.pt',out/'pose.json']}[action]
            # o_voxel imports Triton autotuners even for its CPU conversion API.
            result.append((name,args,outputs,True))
    result.append(('manifests',[c.PYTHON,'-B','-m',c.MODULE+'.worker','finalize'],
                  [c.DATA/'all.json',c.DATA/'train.json',c.DATA/'val.json',c.DATA/'smoke.json',c.OUT/'codec_summary.json'],False))
    for stage in ['ss','shape']:
        out=c.OUT/'smoke'/stage
        base=[c.PYTHON,'-B','-m',c.MODULE+'.invoke','train','--stage',stage,'--manifest',str(c.DATA/'smoke.json'),
             '--output-dir',str(out),'--model-path',c.models()[stage+'_flow'],'--warmup-steps','1',
             '--save-every','1','--log-every','1','--p-uncond','0','--min-condition-views','2',
             '--max-condition-views','4','--verify-cache-hashes']
        result.append(('smoke_'+stage,base+['--max-steps','2'],[out/'checkpoints/step_000002.pt'],True))
        result.append(('resume_'+stage,base+['--max-steps','3','--resume',str(out/'checkpoints/step_000002.pt')],
                       [out/'checkpoints/step_000003.pt'],True))
    result.append(('complete_sampling',[c.PYTHON,'-B','-m',c.MODULE+'.invoke','infer',
       '--manifest',str(c.DATA/'smoke.json'),'--ss-checkpoint',str(c.OUT/'smoke/ss/checkpoints/step_000003.pt'),
       '--shape-checkpoint',str(c.OUT/'smoke/shape/checkpoints/step_000003.pt'),
       '--weights','raw','--ss-steps','12','--shape-steps','12','--output',str(c.OUT/'smoke/generated.ply')],
       [c.OUT/'smoke/generated.ply'],True))
    return result

def run():
    env=c.environment();os.environ.update(env);c.OUT.mkdir(parents=True,exist_ok=True)
    with (c.OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .prepare import main as prepare
        prepare();ident=identity();key=c.digest(ident)
        c.immutable(c.OUT/'runtime.json',{'identity':key,'runtime':ident})
        queue=stages();pid=os.getpid();active=None
        def status(state,**kw):c.write(c.OUT/'status.json',{'time':c.now(),'status':state,'pid':pid,
            'process_start':Path(f'/proc/{pid}/stat').read_text().split()[21],
            'runtime_identity':key,'stage':active,'total_stages':len(queue),**kw})
        try:
            for idx,(name,args,outputs,needs_gpu) in enumerate(queue):
                active=name;receipt=c.OUT/'stages'/(name+'.json')
                if receipt.exists():
                    old=c.read(receipt)
                    if old['runtime_identity']!=key:raise ValueError('Stage binding changed')
                    for artifact in old['artifacts']:
                        p=Path(artifact['path'])
                        if not p.is_file() or p.stat().st_size!=artifact['bytes']:raise ValueError('Completed artifact missing/size changed: '+str(p))
                    continue
                selected=None
                if needs_gpu:
                    while selected is None:
                        selected=gpu()
                        if selected is None:status('waiting_for_gpu',completed_stages=idx,gpus=os.environ.get('T2_SCENE_GPUS','7'));time.sleep(10)
                childenv=dict(env)
                childenv['CUDA_VISIBLE_DEVICES']=str(selected) if needs_gpu else ''
                log=c.OUT/'logs'/(name+f'_{time.time_ns()}.log');log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w') as stream:
                    stream.write('COMMAND '+jsonline(args)+'\n');stream.flush()
                    child=subprocess.Popen(args,cwd=c.REPO,env=childenv,stdout=stream,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        status('running',gpu=selected,child_pid=child.pid,completed_stages=idx,log=str(log));time.sleep(5)
                    if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; see {log}')
                artifacts=[]
                for p in outputs:
                    if not p.is_file() or not p.stat().st_size:raise ValueError('Missing stage artifact: '+str(p))
                    artifacts.append({'path':str(p),'bytes':p.stat().st_size})
                c.write(receipt,{'time':c.now(),'runtime_identity':key,'artifacts':artifacts,'log':str(log)})
            active=None;status('native_complete_review_required',completed_stages=len(queue),
                message='12 native codecs + 2-stage actual training/resume + full 12/12 sampling; baseline and relation training not yet run')
        except BaseException as e:
            status('failed',error=str(e),log=str(log) if 'log' in locals() else None);raise

def jsonline(x):
    import json
    return json.dumps(x,ensure_ascii=False)

def show():
    p=c.OUT/'status.json'
    if not p.exists():print('尚未启动');return
    s=c.read(p);print(f"[{c.now()}] HSSD TRELLIS.2 Native12\n状态：{s['status']}  阶段：{s.get('stage') or '无活动阶段'}  GPU：{s.get('gpu')}")
    print(f"阶段：{s.get('completed_stages',0)}/{s.get('total_stages',90)}；PID：{s['pid']}；更新：{s['time']}")
    try:
        actual=Path(f"/proc/{s['pid']}/stat").read_text().split()[21]
        print('队列进程身份：'+('匹配' if actual==s.get('process_start') else '不匹配'))
    except OSError:print('队列进程：已结束或当前进程空间不可见')
    objects=c.DATA/'objects'
    print('目标12；已编码512：',len(list(objects.glob('*/shape_512.npz'))),'；已解码：',len(list(objects.glob('*/codec.json'))))
    if s.get('log'):
        print('日志：'+s['log'])
        with Path(s['log']).open('rb') as f:
            f.seek(max(0,Path(s['log']).stat().st_size-3000));tail=f.read().decode(errors='replace').splitlines()
        print('\n'.join(tail[-3:]))
    if s.get('error'):print('错误：'+s['error'])
    if s['status']=='native_complete_review_required':print('原生迁移验收计算结束；待看图审阅，尚不是关系模型效果。')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','status','watch']);a=p.parse_args()
    if a.action=='run':run()
    elif a.action=='status':show()
    else:
        while True:
            show();time.sleep(10)

if __name__=='__main__':main()
