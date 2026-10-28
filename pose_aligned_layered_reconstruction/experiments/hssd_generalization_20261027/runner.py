"""Single-worker resumable data qualification, with a lightweight monitor."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_hssd192_common',HERE/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
STOP=False


def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except (OSError,IndexError):return None


def alive(state):
    pid=state.get('pid');start=state.get('process_start_ticks')
    return bool(pid and start and ticks(pid)==start)


def frozen_runtime():
    # The reviewed old runtime binds the vendor renderer, target encoder,
    # coordinate utilities and numerical decoder repairs; retain those sources.
    old=c.read(c.DATA/'outputs/hssd_depth_confidence_20261026/evaluation_identity.json')
    sources=[c.binding(p) for p in sorted(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'PROTOCOL.md']]
    inherited={r['path']:r for r in old['sources']+old['inherited_sources']}
    extra=[c.ROOT/'datasets/scene_occlusion_pilot_20261020/render_tools'/n for n in
           ('prepare_hssd_render.py','render_hssd_blender.py','decode_ktx2.cjs')]
    for p in extra:inherited[str(p)]=c.binding(p)
    for row in inherited.values():c.verify(row)
    value={'sources':sources,'inherited_sources':list(inherited.values()),'repo':c.REPO,'revision':c.REVISION,
           'counts':{'train':128,'val':32,'test':32},'scope':'data qualification; no automatic model training'}
    c.locked(c.OUT/'runtime.json',value);return c.digest(value)


def gpu_inventory():
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],
                         text=True,capture_output=True,check=True)
    compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                           text=True,capture_output=True,check=True)
    pids={}
    for line in compute.stdout.splitlines():
        parts=[s.strip() for s in line.split(',')]
        if len(parts)==2 and parts[1].isdigit():pids.setdefault(parts[0],[]).append(int(parts[1]))
    values=[]
    for line in query.stdout.splitlines():
        index,uuid,used,free=[s.strip() for s in line.split(',')]
        values.append({'gpu':index,'uuid':uuid,'used_mib':int(used),'free_mib':int(free),'compute_pids':pids.get(uuid,[])})
    return values


def choose_gpu(values,request,allow):
    pool=[r for r in values if r['gpu'] in allow and (request=='auto' or r['gpu']==request)]
    idle=[r for r in pool if r['used_mib']<256 and not r['compute_pids']]
    adequate=[r for r in pool if r['free_mib']>=18432]
    available=idle or adequate
    return max(available,key=lambda r:(r['free_mib'],-int(r['gpu']))) if available else None


class Queue:
    def __init__(self,identity):
        self.identity=identity;self.pid=os.getpid();self.start=ticks(self.pid);self.child=None
        self.request=os.environ.get('HSSD_GENERALIZATION_GPU','auto')
        self.allow=os.environ.get('HSSD_GENERALIZATION_GPU_ALLOWLIST','0,1,2,3,4,5,6,7').split(',')

    def status(self,status,stage=None,**kw):
        c.write(c.OUT/'status.json',{'time':c.now(),'status':status,'stage':stage,'pid':self.pid,
                                  'process_start_ticks':self.start,'identity':self.identity,**kw})

    def stage(self,name,command,gpu=False):
        global STOP
        receipt=c.OUT/'stages'/f'{name}.json'
        contract={'identity':self.identity,'command':command,'gpu_stage':gpu}
        if receipt.exists():
            old=c.read(receipt)
            if old['contract']!=contract or old['exit_status']!=0:raise ValueError('Changed completed stage: '+name)
            return
        selected=None
        while gpu and selected is None:
            if STOP:raise InterruptedError('Queue stopped')
            values=gpu_inventory();selected=choose_gpu(values,self.request,self.allow)
            if selected is None:
                self.status('waiting_for_gpu',name,gpu_request=self.request,gpu_inventory=values)
                time.sleep(5)
        env=c.environment()
        if selected:env['CUDA_VISIBLE_DEVICES']=selected['gpu']
        else:env['CUDA_VISIBLE_DEVICES']=''
        log=c.OUT/'logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            self.child=subprocess.Popen(command,cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            while self.child.poll() is None:
                self.status('running',name,child_pid=self.child.pid,log=str(log),gpu=selected['gpu'] if selected else None)
                if STOP:self.child.terminate()
                time.sleep(2)
            code=self.child.returncode;self.child=None
        if STOP:raise InterruptedError('Queue stopped')
        if code!=0:
            self.status('failed',name,log=str(log),error=f'{name} exited {code}; later stages not launched')
            raise RuntimeError(name+' failed; see '+str(log))
        c.write(receipt,{'time':c.now(),'contract':contract,'exit_status':0,'log':str(log)})

    def run(self):
        def py(module,*args):return [c.PYTHON,'-B','-u','-m',c.MODULE+'.'+module,*map(str,args)]
        self.stage('catalog',py('catalog','inventory'))
        self.stage('split',py('split'))
        self.stage('preflight',py('preflight','--write'))
        self.stage('encoder_storage',py('storage'))
        self.stage('download_geometry',py('catalog','assets'))
        self.stage('convert_glb',py('materialize','convert'))
        self.stage('canonical_assets',py('materialize','assets'))
        plan=c.plan()['plan'];scenes=[sid for ids in plan['scene_splits'].values() for sid in ids]
        shared_renderer=c.ROOT/'pose_aligned_layered_reconstruction/experiments/hssd_shared_scene_20261025/render.py'
        for sid in scenes:
            self.stage('assemble_'+sid,[c.BLENDER,'--background','--python',str(HERE/'scene_capture.py'),'--','--scene',sid])
            self.stage('photographs_'+sid,[c.BLENDER,'--background','--python',str(shared_renderer),'--','--output',str(c.DATASET/'scenes'/sid)],gpu=True)
        self.stage('select_targets',py('materialize','select'))
        targets=c.read(c.DATASET/'targets.json')['plan']['objects']
        # Three fixed new assets, one per split, first pass the complete codec
        # chain. Their results are not used to select replacement objects.
        smoke=[next(r for r in targets if r['split']==subset) for subset in ('train','val','test')]
        order=smoke+[r for r in targets if r['uid'] not in {x['uid'] for x in smoke}]
        render=c.ROOT/'pose_aligned_layered_reconstruction/experiments/hssd_pilot_20261020/render_targets.py'
        for obj in order:
            uid=obj['uid'];asset=c.DATASET/'objects'/uid/'asset.json'
            self.stage('target_render_'+uid,[c.BLENDER,'--background','--python',str(render),'--','--asset',str(asset)],gpu=True)
            self.stage('target_latent_'+uid,py('qualify','encode','--uid',uid),gpu=True)
            self.stage('target_visual_'+uid,py('qualify','visual','--uid',uid))
        self.stage('summary',py('qualify','summary'))
        self.status('data_qualification_complete_pending_review',training_launched=False)


def worker():
    global STOP
    c.OUT.mkdir(parents=True,exist_ok=True)
    with (c.OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=Queue(frozen_runtime())
        def handler(*_):
            global STOP;STOP=True
        signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
        try:q.run()
        except InterruptedError:q.status('stopped');return
        except Exception as error:
            state=c.read(c.OUT/'status.json') if (c.OUT/'status.json').exists() else {}
            if state.get('status')!='failed':q.status('failed',state.get('stage'),error=str(error),log=state.get('log'))
            raise


def status():
    if not (c.OUT/'status.json').exists():print('尚未启动数据准备队列。');return
    row=c.read(c.OUT/'status.json');print(f'[{c.now()}] HSSD Generalization192')
    print(f"状态：{row['status']}  阶段：{row.get('stage') or '无活动阶段'}  GPU：{row.get('gpu')}")
    print(f"队列PID：{row.get('pid')}；进程身份：{'匹配' if alive(row) else '已结束或当前进程空间不可见'}")
    if (c.OUT/'progress.json').exists():print('进度：'+json.dumps(c.read(c.OUT/'progress.json'),ensure_ascii=False))
    if (c.OUT/'candidate_plan.json').exists():
        p=c.plan()['plan'];print('场景数：'+str({k:len(v) for k,v in p['scene_splits'].items()})+'；目标预算：'+str(p['target_counts']))
    if (c.DATASET/'targets.json').exists():
        objs=c.read(c.DATASET/'targets.json')['plan']['objects']
        encoded=sum((c.DATASET/'objects'/r['uid']/'latent_audit.json').exists() for r in objs)
        print(f'冻结目标：{len(objs)}/192；已编码：{encoded}/192')
    print('数据根目录：'+str(c.DATASET))
    if row.get('log'):print('日志：'+row['log'])
    if row.get('error'):print('错误：'+row['error'])
    if row['status']=='data_qualification_complete_pending_review':
        print('数据准备与编码结束，等待质量审阅；没有自动启动正式训练。')


def start():
    c.OUT.mkdir(parents=True,exist_ok=True)
    with (c.OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (c.OUT/'status.json').exists() and alive(c.read(c.OUT/'status.json')):status();return
        cpu=c.read(c.OUT/'preflight.json');blender=c.read(c.OUT/'blender_math_smoke.json')
        if not cpu['passed'] or not blender['passed']:raise ValueError('Complete the CPU/Blender smoke first')
        c.verify(cpu['source'])
        for source in blender['sources']:c.verify(source)
        identity=frozen_runtime();log=c.OUT/f'launcher_{time.time_ns()}.log'
        with log.open('w') as stream:
            proc=subprocess.Popen([c.PYTHON,'-B','-u',str(__file__),'worker'],cwd=c.ROOT,env=c.environment(),
                                  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        c.write(c.OUT/'status.json',{'time':c.now(),'status':'starting','stage':None,'pid':proc.pid,
                                  'process_start_ticks':ticks(proc.pid),'identity':identity,'launcher':str(log)})
        print(json.dumps({'started':True,'pid':proc.pid,'launcher':str(log),'data':str(c.DATASET)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start-prepare','status','watch','stop','worker','smoke'])
    args=parser.parse_args()
    if args.action=='start-prepare':start()
    elif args.action=='worker':worker()
    elif args.action=='status':status()
    elif args.action=='smoke':
        env=c.environment();env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([c.PYTHON,'-B','-m',c.MODULE+'.preflight','--write'],cwd=c.ROOT,env=env,check=True)
        subprocess.run([c.BLENDER,'--background','--factory-startup','--python',str(HERE/'blender_math_smoke.py')],cwd=c.ROOT,env=env,check=True)
    elif args.action=='stop':
        row=c.read(c.OUT/'status.json')
        if alive(row):os.kill(row['pid'],signal.SIGTERM);print('已请求停止本数据准备队列。')
        else:print('没有身份匹配的活动队列。')
    else:
        try:
            while True:status();time.sleep(10)
        except KeyboardInterrupt:pass
