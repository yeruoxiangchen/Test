"""Operational GPU handoff only; frozen numerical runtime stays unchanged."""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
RUNNER=HERE.parent/'runner.py'
sys.path.insert(0,str(HERE.parent))
spec=importlib.util.spec_from_file_location('_oracle_operational_runner',RUNNER)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
import common


def gpu_memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    result={}
    for line in raw.strip().splitlines():
        index,free,used=[x.strip() for x in line.split(',')]
        if not all(v.isdigit() for v in (index,free,used)):raise ValueError('Invalid GPU memory inventory')
        result[index]={'free_mib':int(free),'used_mib':int(used)}
    return result


def ready(info,minimum):
    return info['free_mib']>=minimum


def verify_completed():
    identity=common.freeze_runtime();common.check_smoke()
    completed=common.read(common.train_dir(common.ARMS[0])/'complete.json')
    if completed['step']!=200:raise ValueError('Expected finished 200-step training')
    common.verify(completed['checkpoint'])
    results=[]
    for case in common.new_jobs():
        path=common.case_dir(case)/'ss.json';row=common.read(path)
        if (not row['success'] or row['identity']!=identity or row['case']!=case
                or row['trained_model']['checkpoint']!=completed['checkpoint']):
            raise ValueError('Incomplete or mismatched SS result')
        common.verify(row['coords']);results.append(common.binding(path))
    if len(results)!=12:raise ValueError('Expected exactly 12 SS results')
    return identity,common.binding(common.train_dir(common.ARMS[0])/'complete.json'),results


def work(args):
    with runner.open_lock(args.lock_fd):
        runner.SELECTED_GPU=str(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        os.environ['HSSD_VISIBLE_ORACLE_GPU']=str(args.gpu)
        runner.state({'status':'running','stage':'resume_verification'})
        try:
            identity,training,results=verify_completed()
            amendment={'time':common.now(),'kind':'operational_gpu_handoff',
                'runtime_identity':identity,'numeric_runtime_unchanged':True,
                'gpu':str(args.gpu),'allow_shared':True,'minimum_free_mib':args.min_free_mib,
                'training_complete':training,'ss_results':results,
                'scripts':[common.binding(__file__),common.binding(HERE/'resume_mesh.sh')],
                'scope':'reuse completed training/SS/smoke; only mesh then summary; retain other GPU processes',
                'memory_evidence':'smoke process peak 7587.43 MiB; new supports up to 32450 voxels; no guaranteed all-case peak'}
            record=common.OUT/'operations'/f'gpu_resume_{time.time_ns()}.json'
            common.write(record,amendment)
            while True:
                memory=gpu_memory()
                info=memory[str(args.gpu)]
                if ready(info,args.min_free_mib):break
                wait_log=common.OUT/'logs/gpu_resume_wait.log'
                wait_log.parent.mkdir(parents=True,exist_ok=True)
                with wait_log.open('a') as stream:
                    stream.write(f'{common.now()} GPU {args.gpu}: free={info["free_mib"]}MiB; '+
                        f'waiting for >={args.min_free_mib}MiB; shared GPU permitted\n')
                runner.state({'status':'running','stage':'waiting_shared_gpu_memory',
                    'log':str(wait_log),'reason':'Shared-GPU resume waiting for free-memory threshold',
                    'minimum_free_mib':args.min_free_mib,'gpu_memory':memory,
                    'operational_amendment':common.binding(record)})
                common.event('resume_waiting_memory',gpu=args.gpu,**info,minimum_free_mib=args.min_free_mib)
                time.sleep(20)
            common.event('resume_gpu_selected',gpu=args.gpu,**info,allow_shared=True,
                operational_amendment=common.binding(record))
            # Explicitly bypass ONLY original exclusive-card scheduling. The
            # original stage timeout/heartbeat/cleanup and numerical modules run unchanged.
            runner.stage('mesh_geometry_oracle_visible','evaluate','mesh','--arm',common.ARMS[0],
                timeout=86400,uses_gpu=False)
            runner.stage('summary','summarize',timeout=14400)
            runner.state({'status':'all_complete_review_required','stage':None,
                'operational_amendment':common.binding(record)})
            common.event('queue',phase='complete',action='mesh_resume_and_summary')
        except BaseException as error:
            previous=common.read(common.OUT/'status.json')
            runner.state({**previous,'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                'child_pid':None,'error':str(error) or type(error).__name__})
            raise


def start(args):
    # Use the same queue lock; refuses to start alongside the original queue.
    with runner.open_lock() as lock:
        log=common.OUT/f'gpu_resume_launcher_{time.time_ns()}.log'
        cmd=[str(common.PYTHON),str(Path(__file__).resolve()),'worker','--gpu',str(args.gpu),
            '--min-free-mib',str(args.min_free_mib),'--runner',str(RUNNER),
            '--lock-fd',str(lock.fileno())]
        with log.open('w') as stream:
            process=subprocess.Popen(cmd,cwd=str(common.ROOT),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        common.write(common.OUT/'operations'/f'launch_{time.time_ns()}.json',{
            'time':common.now(),'pid':process.pid,'process_start_ticks':runner.process_start_ticks(process.pid),
            'gpu':str(args.gpu),'log':str(log),'command':cmd,'training_and_ss_reused':True})
        print(f'Mesh resume started: GPU {args.gpu}, PID {process.pid}; log {log}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','worker','self-test'])
    parser.add_argument('--gpu',type=int,choices=range(8),default=2)
    parser.add_argument('--min-free-mib',type=int,default=18432)
    parser.add_argument('--runner',default=str(RUNNER))
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    if Path(args.runner)!=RUNNER:raise ValueError('Unexpected frozen runner path')
    if args.min_free_mib<18432:raise ValueError('Do not reduce the 18 GiB admission threshold')
    if args.action=='self-test':
        assert ready({'free_mib':18432},18432) and not ready({'free_mib':18431},18432)
        assert not runner.own_process(1)
        print('CPU scheduling checks passed; no CUDA or output writes')
        return
    runner.configure_environment()
    signal.signal(signal.SIGTERM,runner.signal_stop)
    if args.action=='start':start(args)
    else:work(args)


if __name__=='__main__':main()
