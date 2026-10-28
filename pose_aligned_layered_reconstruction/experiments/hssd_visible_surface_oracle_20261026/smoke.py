"""Bounded GPU gates before the matched 200-step training queue.

No model is loaded at import time. Short training uses isolated smoke folders
and the same 200-step contract; its checkpoints never seed the formal runs.
"""
import gc
import math
import torch

from .common import (ARMS,STEPS,FORMAT,OUT,plan,policy,freeze_runtime,read,write,
    binding,verify,digest,now,event,tensor_hash,check_smoke)
from .data import load_case
from .model import build
from .train import train
from .evaluate import smoke_replay,tensor_binding,verify_nested
from ...frozen_slat import configure_determinism,bootstrap_external_paths


def release_cuda():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def check_receipt(path,identity,required):
    if not path.exists():
        return None
    row = read(path)
    if row.get('identity')!=identity or not row.get('passed') or not all(row.get(k) is True for k in required):
        raise ValueError('Short-check receipt is failed or belongs to another runtime: '+str(path))
    for item in row.get('artifacts',[]):
        verify(item)
    return row


def reference_runs():
    """Bind the two completed historical controls without loading their weights."""
    refs=plan()['plan']['reference_training']
    result={}
    for arm in ('appearance_control','geometry_observed'):
        row=refs[arm]
        run=read(verify(row['run']));complete=read(verify(row['complete']))
        if complete['step']!=STEPS or run['contract']!=complete['contract']:
            raise ValueError('Historical comparison is incomplete: '+arm)
        if run['data_identity']!=complete['data_identity']:
            raise ValueError('Historical training data identity differs: '+arm)
        current=policy(ARMS[0])
        for key in ('steps','seed','lr','warmup','betas','weight_decay','batch','p_uncond',
                    'weights','amp','residual_channels','projection_bias','parent_checkpoint',
                    'observation_variant','flow_loss','view_selection'):
            if run['contract'][key]!=current[key]:
                raise ValueError('Oracle/historical numerical training policy differs: '+key)
        result[arm]=run
    return result


def initial_checks(identity):
    path=OUT/'smoke/init_check.json'
    previous=check_receipt(path,identity,('zero_initialization_exact','matched_trainable_capacity',
        'parent_grad_none','new_projection_gradient_nonzero'))
    if previous is not None:
        verify_nested(previous['input_sources']);verify(previous['parent_checkpoint'])
        return binding(path)
    refs=reference_runs();p=plan()['plan'];arm=ARMS[0];contract=policy(arm)
    device=torch.device('cuda:0')
    obj=next(o for o in p['objects'] if o['uid'].endswith('i075'))
    event('smoke_initialization',phase='load_complete_old_parent')
    sampler,model,saved,bindings=build(contract,device,checkpointing=True)
    del sampler,saved
    model.eval();parent=model.parent
    versions={name:q._version for name,q in parent.named_parameters()}
    init=tensor_hash(model.residual.state_dict())
    count=sum(q.numel() for q in model.parameters() if q.requires_grad)
    if count!=3164160 or count!=sum(q.numel() for q in model.residual.parameters()):
        raise ValueError('Unexpected trainable oracle capacity')
    for old_arm,run in refs.items():
        if (run['initial_residual_sha']!=init or run['trainable_parameter_count']!=count
                or run['bindings']!=bindings):
            raise ValueError('Initialization or complete parent differs from '+old_arm)
    _,sample,inputs,evidence,eb=load_case(obj,arm,{})
    for old_arm,run in refs.items():
        old_input=next(r for r in run['data_identity'] if r['uid']==obj['uid'])
        if old_input['inputs']!=inputs:
            raise ValueError('Zero gate image inputs differ from '+old_arm)
    generator=torch.Generator(device=device).manual_seed(contract['seed']+1229)
    x=torch.randn((1,8,16,16,16),device=device,generator=generator)
    t=torch.tensor([500.],device=device);context=sample['stock_condition'].to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prepared=model.prepare(sample,device,evidence)
        with torch.no_grad():reference=model(x,t,context,prepared,enabled=False)
        prediction=model(x,t,context,prepared,enabled=True)
        if not torch.equal(reference,prediction):
            raise ValueError('Initial oracle changes complete parent forward')
        parent_output=tensor_binding(reference)
        loss=prediction.float().square().mean()
    artifacts=[b for row in p['reference_training'].values() for b in row.values()]
    if p.get('reference_initial_check'):
        old_initial=read(verify(p['reference_initial_check']))
        if (old_initial['initial_residual_sha']!=init or old_initial['trainable_parameter_count']!=count
                or any(r['parent_output']!=parent_output for r in old_initial['rows'])):
            raise ValueError('Zero gate differs from historical CUDA parent forward')
        artifacts.append(p['reference_initial_check'])
    if not torch.isfinite(loss):raise ValueError('Nonfinite initialization objective')
    model.zero_grad(set_to_none=True);loss.backward()
    parent_clean=all(q.grad is None and q._version==versions[name] for name,q in parent.named_parameters())
    gradients=[q.grad for q in model.residual.projections.parameters()]
    nonzero=any(g is not None and bool(torch.any(g!=0)) for g in gradients)
    finite=all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
    if not parent_clean or not nonzero or not finite:
        raise ValueError('Initial oracle gradient isolation failed')
    if tensor_hash(model.residual.state_dict())!=init:
        raise ValueError('Gradient check unexpectedly updated oracle')
    row={'arm':arm,'contract':contract,'inputs':inputs,'evidence_binding':eb,
        'initial_residual_sha':init,'trainable_parameter_count':count,
        'parent_output':parent_output,'new_output':tensor_binding(prediction),
        'zero_injection_exact':True,'parent_grad_none':True,'parent_versions_unchanged':True,
        'new_projection_gradient_nonzero':True,'new_projection_gradients_finite':True,
        'gradient_norm':float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients if g is not None))),
        'optimizer_steps':0,'objective':float(loss.detach())}
    model.zero_grad(set_to_none=True)
    del model,parent,prepared,reference,prediction,loss,context,gradients,x,t
    release_cuda()
    write(path,{'time':now(),'identity':identity,'passed':True,
        'zero_initialization_exact':True,'matched_trainable_capacity':True,
        'parent_grad_none':True,'new_projection_gradient_nonzero':True,
        'initial_residual_sha':init,'trainable_parameter_count':count,
        'parent_checkpoint':contract['parent_checkpoint'],'parent_binding':bindings,
        'rows':[row],'input_sources':{arm:{'inputs':inputs,'evidence_binding':eb}},
        'artifacts':artifacts,'scope':'actual table2072 CUDA forward/gradient and historical initialization; no optimizer update'})
    return binding(path)


def short_training_checks(identity,initial_binding):
    path=OUT/'smoke/training_check.json'
    previous=check_receipt(path,identity,('two_step_training_passed','shared_noise_order_exact',
        'matched_trainable_capacity','same_initial_residual'))
    if previous is not None:
        if previous['initial_check']!=initial_binding:
            raise ValueError('Short training was bound to a different initial gate')
        return binding(path)
    import json
    initial=read(verify(initial_binding));p=plan()['plan'];arm=ARMS[0]
    folder=OUT/'smoke'/('train_'+arm)
    event('smoke_two_step_training',phase='starting',arm=arm)
    train(arm,steps=STEPS,output=folder,max_steps=2);release_cuda()
    if (folder/'complete.json').exists():
        raise ValueError('Two-step smoke must not masquerade as completed formal training')
    run=read(folder/'run.json');partial=read(folder/'partial.json')
    checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    expected=policy(arm)
    if (run['contract']!=expected or checkpoint['contract']!=expected or partial['contract']!=expected
            or checkpoint['format']!=FORMAT or checkpoint['step']!=2 or partial['step']!=2
            or partial['planned_steps']!=STEPS or partial['checkpoint']!=binding(folder/'last.pt')
            or checkpoint['bindings']!=run['bindings'] or checkpoint['data_identity']!=run['data_identity']
            or checkpoint['initial_residual_sha']!=run['initial_residual_sha']
            or checkpoint.get('frozen_parameter_versions_unchanged') is not True):
        raise ValueError('Short checkpoint/run/partial contract mismatch')
    if (run['initial_residual_sha']!=initial['initial_residual_sha']
            or run['trainable_parameter_count']!=initial['trainable_parameter_count']):
        raise ValueError('Training initialization/capacity differs from zero gate')
    if tensor_hash(checkpoint['residual'])==run['initial_residual_sha']:
        raise ValueError('Short training did not update the residual')
    history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [r['step'] for r in history]!=[1,2]:
        raise ValueError('Short training must contain exactly two steps')
    for row in history:
        if (row['arm']!=arm or any(not math.isfinite(row[key]) for key in
                ('flow_loss','gradient_norm','diffusion_time','learning_rate'))
                or row['gradient_norm']<=0 or len(row['noise_sha'])!=64):
            raise ValueError('Invalid short training loss/gradient/noise record')
    fields=('step','uid','index','original_index','epoch','offset','diffusion_time','noise_sha','learning_rate','object_group_exposures')
    paired=[{key:r[key] for key in fields} for r in history]
    artifacts=[initial_binding]+[binding(folder/name) for name in ('run.json','history.jsonl','last.pt','partial.json')]
    for old_arm,old_run in reference_runs().items():
        rb=p['reference_training'][old_arm]
        old_history=[json.loads(line) for line in verify(rb['history']).read_text().splitlines()]
        if [r['step'] for r in old_history]!=list(range(1,STEPS+1)):
            raise ValueError('Historical complete schedule missing: '+old_arm)
        if paired!=[{key:r[key] for key in fields} for r in old_history[:2]]:
            raise ValueError('Oracle changed old batch/time/noise/order: '+old_arm)
        if len(run['data_identity'])!=len(old_run['data_identity']):
            raise ValueError('Changed training object cohort')
        for a,b in zip(run['data_identity'],old_run['data_identity']):
            if any(a[key]!=b[key] for key in ('uid','original_index','inputs','target_source','target_tensor_sha')):
                raise ValueError('Oracle changed images or target: '+old_arm)
        artifacts.extend(rb.values())
    if p.get('previous_smoke_training_check'):
        old_check=read(verify(p['previous_smoke_training_check']))
        if not old_check['passed'] or paired!=old_check['paired_schedule']:
            raise ValueError('Oracle changed archived two-step smoke schedule')
        artifacts.append(p['previous_smoke_training_check'])
    write(path,{'time':now(),'identity':identity,'passed':True,'initial_check':initial_binding,
        'two_step_training_passed':True,'shared_noise_order_exact':True,'matched_trainable_capacity':True,
        'same_initial_residual':True,'planned_formal_steps':STEPS,'short_steps':2,'paired_schedule':paired,
        'initial_residual_sha':initial['initial_residual_sha'],'trainable_parameter_count':initial['trainable_parameter_count'],
        'histories':{arm:history},'artifacts':artifacts,
        'scope':'isolated 2-step oracle; exact historical schedule/input/target; formal 200-step starts fresh'})
    return binding(path)


def main():
    configure_determinism();bootstrap_external_paths();torch.set_num_threads(2)
    identity=freeze_runtime()
    if (OUT/'smoke.json').exists():
        check_smoke()
        event('smoke_complete',reused=True)
        return
    initial=initial_checks(identity)
    release_cuda()
    replay=smoke_replay()
    release_cuda()
    replay_row=read(verify(replay))
    if (replay_row['identity']!=identity or not replay_row['passed']
            or not replay_row['ss_exact'] or not replay_row['mesh_exact']):
        raise ValueError('Full parent SS/mesh replay did not pass')
    trained=short_training_checks(identity,initial)
    release_cuda()
    train_row=read(verify(trained))
    artifacts=[initial,replay,trained,*train_row['artifacts']]
    artifacts += [replay_row['ss_record'],replay_row['mesh_record']]
    replay_ss=read(verify(replay_row['ss_record']))
    replay_mesh=read(verify(replay_row['mesh_record']))
    artifacts += [replay_ss['coords'],replay_mesh['mesh'],replay_mesh['world_mesh'],
                  replay_mesh['sampled_slat_receipt']]
    latent_receipt=read(verify(replay_mesh['sampled_slat_receipt']))
    artifacts += [latent_receipt['latent']]
    # The final gate binds generated arrays as well as their metadata, so a
    # cached JSON receipt cannot hide a later mesh/latent/coordinate change.
    artifacts=list({str(item['path']):item for item in artifacts}.values())
    for item in artifacts:verify(item)
    write(OUT/'smoke.json',{'time':now(),'identity':identity,'passed':True,
        'parent_ss_mesh_replay_exact':True,'zero_initialization_exact':True,
        'matched_trainable_capacity':True,'two_step_training_passed':True,'shared_noise_order_exact':True,
        'initial_check':initial,'replay':replay,'training_check':trained,
        'artifacts':artifacts,'formal_training_started':False,
        'scope':'pipeline/initialization/trainability checks only; no method quality claim'})
    event('smoke_complete',parent_replay=True,matched_initialization=True,two_step_training=True)


if __name__=='__main__':
    main()
