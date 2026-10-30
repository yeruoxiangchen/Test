"""Second controlled experiment: only the Shape repair trajectory is changed."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
from . import common as c
from . import runner as queue

V1=c.ROOT/'outputs/round1_20261029'
OUT=c.ROOT/'outputs/bridge_v2_20261030'
UNIT='pose-conditioned-geometry-repair-bridge-v2-20261030'
SS=V1/'overfit_repair_ss/checkpoints/step_000600.pt'


def configure():
    c.OUT=OUT


def pipeline():
    configure()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.getpid()
    start=Path(f'/proc/{pid}/stat').read_text().split()[21]
    def phase(name,state='running',**kw):
        c.write(OUT/'status.json',{'time':c.now(),'state':state,'phase':name,
            'pid':pid,'process_start':start,**kw})
    try:
        c.write(OUT/'protocol.json',{'time':c.now(),'plan_identity':c.plan()['identity'],
            'change':'Shape denoising FM -> damaged-to-clean residual bridge',
            'ss_unchanged':str(SS),'stock_reference':'adapted flow minus frozen official flow',
            'bridge_input':'(1-t)*clean+t*damaged+0.1*t*noise',
            'velocity_target':'damaged-clean; known noise augmentation excluded from integration',
            'native_time':'650*t','train_endpoint_probability':.5,
            'baseline_correction':'actual input omits absent Shape coordinates; teacher-support baseline remains separate',
            'no_retroactive_pass':'Round1 results and gate are preserved',
            'steps':[600,2000],'formal_training':'automatically follows the amended overfit gate; fresh official initialization'})
        phase('cpu_checks')
        with (OUT/'tests.log').open('w') as f:
            subprocess.run(['bash',str(c.HERE/'run.sh'),'tests'],env=c.environment(),
                cwd=c.REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
        phase('physical_damaged_input_baseline')
        queue.command('baseline_audit','baseline_audit',['--output',OUT/'baseline_audit'],
            gpu=True,required=[OUT/'baseline_audit/summary.json'])
        phase('bridge_gpu_smoke')
        smoke=OUT/'smoke_shape'
        args=['--stage','shape','--subset','overfit','--objective','bridge',
              '--output',smoke,'--steps',3,'--save-every',3,'--log-every',1,'--warmup',1]
        queue.command('bridge_smoke','train',args,gpu=True,required=[smoke/'checkpoints/step_000003.pt'])
        args[args.index('--steps')+1]=5
        args+=['--resume',smoke/'checkpoints/step_000003.pt']
        queue.command('bridge_resume','train',args,gpu=True,required=[smoke/'checkpoints/step_000005.pt'])
        phase('bridge_native_evaluation_smoke')
        queue.command('bridge_evaluation_smoke','evaluate',['--stage','shape','--subset','overfit',
            '--objective','bridge','--checkpoint',smoke/'checkpoints/step_000005.pt',
            '--ss-checkpoint',SS,'--output',OUT/'evaluation_smoke','--limit',1],
            gpu=True,required=[OUT/'evaluation_smoke/summary.json'])
        for steps in [600,2000]:
            phase('overfit32_bridge_shape',target_steps=steps)
            output=OUT/'overfit_repair_shape'
            args=['--stage','shape','--subset','overfit','--objective','bridge',
                  '--output',output,'--steps',steps,'--save-every',200,'--log-every',10,'--warmup',30]
            last=output/'checkpoints/last.pt'
            if last.exists():
                args+=['--resume',last]
            ck=output/'checkpoints'/f'step_{steps:06d}.pt'
            queue.command('bridge_train_'+str(steps),'train',args,gpu=True,
                required=[ck,output/'complete.json'])
            phase('evaluate_bridge32',target_steps=steps)
            evaluation=OUT/f'eval_step{steps}'
            queue.command('bridge_eval_'+str(steps),'evaluate',['--stage','shape','--subset','overfit',
                '--objective','bridge','--checkpoint',ck,'--ss-checkpoint',SS,
                '--output',evaluation,'--limit',32],gpu=True,required=[evaluation/'summary.json'])
            summary=c.read(evaluation/'summary.json')
            if summary['overfit_gate_passed']:
                c.write(OUT/'overfit_gate.json',{'time':c.now(),'passed':True,'shape':summary,
                    'ss_checkpoint':str(SS),'shape_checkpoint':str(ck),
                    'scope':'32 training objects, known missing ROI, SS held fixed; not Dev64 generalization'})
                phase('prepare_formal2000_dev64')
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(queue.command,'prepare_formal_rank'+str(r),'prepare',
                        ['objects','--subset','all','--rank',r,'--world',2],True) for r in range(2)]
                    for f in futures:
                        f.result()
                for subset in ['train','dev']:
                    queue.command('publish_'+subset,'prepare',['publish','--subset',subset],
                        required=[c.DATA/f'{subset}.json'])
                phase('formal_training')
                def train_formal(arm,stage):
                    objective='bridge' if arm=='repair' and stage=='shape' else 'flow'
                    folder=OUT/f'train_{arm}_{stage}'
                    args=['--stage',stage,'--arm',arm,'--subset','train','--objective',objective,
                        '--output',folder,'--steps',4000,'--save-every',200,'--log-every',10,
                        '--warmup',100,'--grad-accum',4]
                    if (folder/'checkpoints/last.pt').exists():
                        args+=['--resume',folder/'checkpoints/last.pt']
                    checkpoint=folder/'checkpoints/step_004000.pt'
                    queue.command(f'formal_{arm}_{stage}','train',args,True,required=[checkpoint,folder/'complete.json'])
                    return checkpoint
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures={(arm,stage):pool.submit(train_formal,arm,stage)
                        for arm in ['repair','full'] for stage in ['ss','shape']}
                    checkpoints={k:f.result() for k,f in futures.items()}
                phase('formal_dev64_evaluation')
                summaries={}
                for arm in ['repair','full']:
                    for seed in [42,43]:
                        objective='bridge' if arm=='repair' else 'flow'
                        folder=OUT/f'dev64_{arm}_s{seed}'
                        queue.command(f'dev64_{arm}_s{seed}','evaluate',['--stage','shape','--arm',arm,
                            '--subset','dev','--objective',objective,'--checkpoint',checkpoints[arm,'shape'],
                            '--ss-checkpoint',checkpoints[arm,'ss'],'--output',folder,'--limit',64,
                            '--seed',seed,'--variants',0,1,2,3],True,required=[folder/'summary.json'])
                        summaries[f'{arm}_{seed}']=c.read(folder/'summary.json')
                c.write(OUT/'formal_summary.json',{'time':c.now(),'complete':True,'summaries':summaries,
                    'scope':'Known synthetic ROI, held-out UIDs; automatic localization and real generated errors remain separate tests'})
                phase('complete','complete')
                return
        phase('overfit_gate_failed','failed',error='Residual bridge did not pass the corrected 32-object gate')
    except Exception as e:
        phase('failed','failed',error=str(e))
        raise
    finally:
        lock.close()


def launch():
    configure()
    p=OUT/'status.json'
    if p.exists():
        status=c.read(p)
        if status['state']=='running' and queue.alive(status['pid'],status['process_start']):
            print('Bridge queue already running')
            return
    subprocess.run(['systemd-run','--user','--unit='+UNIT,'--collect',
        '--property=Nice=10','--property=IOSchedulingClass=best-effort',
        '--property=IOSchedulingPriority=7','--property=WorkingDirectory='+str(c.REPO),
        '--setenv=T2_REPAIR_GPUS='+os.environ.get('T2_REPAIR_GPUS','3,2,5,6,7'),
        '/bin/bash',str(c.HERE/'run.sh'),'bridge-pipeline'],env=c.environment(),check=True)
    print('Started',UNIT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['launch','pipeline','status'])
    a=p.parse_args()
    configure()
    if a.action=='status':
        queue.status()
    else:
        {'launch':launch,'pipeline':pipeline}[a.action]()


if __name__=='__main__':
    main()
