#!/usr/bin/env python3
"""Bounded background repair evaluations with a frozen runtime and local GPU API."""
import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from screenshot_agent.config import load_config
from screenshot_agent.control import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source():
    state=read(ROOT/'source_state.json')
    changed=[name for name,value in state['sha256'].items() if digest(ROOT/name)!=value]
    if changed:
        raise ValueError(f'Frozen source changed: {changed}')
    return state


def status(output):
    result={}
    for name in ['ablation','extended']:
        folder=output/name
        if not (folder/'status.json').exists():
            result[name]={'status':'not_started'}
            continue
        item=read(folder/'status.json')
        completed,total,resolved=0,0,0
        phases={}
        for row in item.get('phases',[]):
            phase=folder/row['name']/'evaluation'
            scheduled=row['runs']
            progress=read(phase/'status.json') if (phase/'status.json').exists() else {'status':'pending','completed':0,'total':scheduled}
            phases[row['name']]={k:progress[k] for k in ['status','completed','total','current'] if k in progress}
            completed+=progress['completed'];total+=scheduled
            for f in (phase/'runs').glob('*/result.json'):
                resolved+=bool(read(f)['resolved'])
        result[name]={k:v for k,v in item.items() if k not in ['phases','source']}
        result[name].update(completed=completed,total=total,resolved_so_far=resolved,phases=phases)
    return result


def execute(mode,output,gpu,port):
    source=verify_source()
    output.mkdir(parents=True,exist_ok=False)
    phases=[]
    if mode=='ablation':
        data=ROOT/'datasets/generated/tool_ablation_v5'
        for variant in ['clean','degraded']:
            for policy in ['available','initial_only']:
                phases.append({'name':variant+'_'+policy,'manifest':str((data/(variant+'.jsonl')).resolve()),
                    'readers':'vlm','policy':policy,'runs':20})
    else:
        data=ROOT/'datasets/generated/swebench_extended_v5'
        preflight=read(data/'build_status.json')
        build=read(data/'build_report.json')
        if preflight['completed']!=60 or build['evaluation_tasks']!=50 or build['evaluation_repositories']<5:
            raise ValueError('Wait for all 60 preflights and a fixed 50-task, >=5-repository manifest')
        phases=[{'name':'three_readers','manifest':str((data/'tasks.jsonl').resolve()),
                 'readers':'oracle,ocr,vlm','policy':'available','runs':150}]
    for phase in phases:
        phase['manifest_sha256']=digest(Path(phase['manifest']))
    state={'status':'starting','mode':mode,'gpu':gpu,'port':port,'started_at':time.time(),
        'phases':phases,'source':source,'pilot_excluded_from_primary_scores':mode=='ablation'}
    atomic_json(output/'status.json',state)
    atomic_json(output/'plan.json',state)
    server=active=None
    handles=[]
    def terminate(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
    def cancel(signum,frame):
        raise KeyboardInterrupt('Stop requested')
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,cancel)
    def invoke(argv,log):
        nonlocal active
        stream=log.open('ab');handles.append(stream)
        active=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        rc=active.wait()
        active=None
        if rc:
            raise RuntimeError(f'Command failed with exit {rc}; inspect {log}')
    try:
        used=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>1000:
            raise ValueError(f'GPU {gpu} is occupied ({used} MiB); no processes were stopped')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_HOME=str(ROOT/'.cache/huggingface'),TORCH_HOME=str(ROOT/'.cache/torch'),
            XDG_CACHE_HOME=str(ROOT/'.cache'),TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),
            CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
            NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1',
            PYTHONPATH=str(ROOT/'src'))
        key=secrets.token_urlsafe(32)
        endpoint=f'http://127.0.0.1:{port}/v1'
        env.update(SCA_LOCAL_KEY=key,SCA_CODE_API_KEY=key,SCA_VISION_API_KEY=key,
            SCA_CODE_MODEL='local-qwen3-vl-8b',SCA_VISION_MODEL='local-qwen3-vl-8b',
            SCA_CODE_BASE_URL=endpoint,SCA_VISION_BASE_URL=endpoint)
        log=(output/'model_server.log').open('ab');handles.append(log)
        server=subprocess.Popen([str(ROOT/'.model-env/bin/python'),'-u',str(ROOT/'scripts/local_model_api.py'),
            '--model-path','/data/zjr/models/Qwen3-VL-8B-Instruct','--port',str(port),
            '--max-context','16384','--max-output','2048'],stdout=log,stderr=subprocess.STDOUT,
            env=env,start_new_session=True)
        atomic_json(output/'model_process.json',{'pid':server.pid,'gpu':gpu})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def health():
            with opener.open(f'http://127.0.0.1:{port}/health',timeout=2) as reply:
                return json.load(reply)
        deadline=time.monotonic()+900
        while True:
            if server.poll() is not None:
                raise RuntimeError('Model server exited; inspect model_server.log')
            try:
                ready=health()
                if ready.get('ready'):
                    atomic_json(output/'ready.json',ready)
                    break
            except (OSError,ValueError):
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('Model startup exceeded 900 seconds')
            time.sleep(2)
        # Expand variables only in this process; never serialize credentials.
        for name,value in env.items():
            if name.startswith('SCA_'):
                os.environ[name]=value
        config=load_config(ROOT/'configs/v5-local.json')
        for policy in ['available','initial_only']:
            current=config.model_copy(deep=True)
            current.screenshots.allow_crop=current.screenshots.allow_reread=policy=='available'
            atomic_json(output/(policy+'.json'),current.model_dump())
        if mode=='ablation':
            state.update(status='pilot',current='pilot')
            atomic_json(output/'status.json',state)
            pilot=output/'pilot'
            invoke([sys.executable,'-u','-m','screenshot_agent','evaluate','--manifest',phases[0]['manifest'],
                '--config',str(output/'available.json'),'--readers','vlm','--model','api','--limit','2',
                '--output',str(pilot)],output/'pilot.log')
            rows=read(pilot/'report.json')['records']
            if not any(r['resolved'] and r['usage']['prompt_tokens']>0 for r in rows) or any(
                    r['status']=='error' for r in rows):
                raise RuntimeError('Pilot failed the end-to-end plumbing check; full comparisons were not started')
            atomic_json(output/'pilot_passed.json',{'runs':len(rows),'resolved':sum(r['resolved'] for r in rows),'included_in_primary_scores':False})
        for phase in phases:
            if digest(Path(phase['manifest']))!=phase['manifest_sha256']:
                raise ValueError('Manifest changed after the plan was frozen')
            phase_root=output/phase['name']
            phase_root.mkdir()
            state.update(status='running',current=phase['name'])
            atomic_json(output/'status.json',state)
            print(f"Starting {mode}/{phase['name']}: {phase['runs']} runs on GPU {gpu}",flush=True)
            invoke([sys.executable,'-u','-m','screenshot_agent','evaluate','--manifest',phase['manifest'],
                '--config',str(output/(phase['policy']+'.json')),'--readers',phase['readers'],'--model','api',
                '--output',str(phase_root/'evaluation')],phase_root/'evaluation.log')
            if read(phase_root/'evaluation/status.json')['completed']!=phase['runs']:
                raise ValueError('Incomplete scheduled comparison')
            if mode=='extended':
                invoke([sys.executable,'-u',str(ROOT/'scripts/clean_official_patches.py'),'--run',str(phase_root)],
                    phase_root/'clean_patch.log')
        atomic_json(output/'final_model_stats.json',health())
        verify_source()
        state['status']='complete'
    except KeyboardInterrupt as error:
        state.update(status='cancelled',error=str(error))
    except Exception as error:
        state.update(status='error',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        terminate(active)
        terminate(server)
        for handle in handles:
            handle.close()
        state['finished_at']=time.time()
        state['model_released']=server is None or server.poll() is not None
        atomic_json(output/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['ablation','extended','status'])
    p.add_argument('--output',type=Path,default=Path('outputs/v5_supplementary'))
    p.add_argument('--gpu',default='7')
    p.add_argument('--port',type=int,default=18817)
    a=p.parse_args()
    if a.command=='status':
        print(json.dumps(status(a.output.resolve()),ensure_ascii=False,indent=2))
    else:
        execute(a.command,a.output.resolve()/a.command,a.gpu,a.port)
