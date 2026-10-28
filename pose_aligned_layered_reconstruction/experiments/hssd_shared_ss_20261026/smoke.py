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
from .model import build,SharedEvidenceSS
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


def initial_checks(identity):
    path = OUT/'smoke/init_check.json'
    previous = check_receipt(path,identity,('zero_initialization_exact','matched_trainable_capacity',
        'parent_grad_none','new_projection_gradient_nonzero'))
    if previous is not None:
        verify_nested(previous['input_sources'])
        verify(previous['parent_checkpoint'])
        return binding(path)
    device = torch.device('cuda:0')
    obj = next(o for o in plan()['plan']['objects'] if o['uid'].endswith('i075'))
    contracts = {arm:policy(arm) for arm in ARMS}
    shared_contract = lambda c:{k:v for k,v in c.items() if k!='arm'}
    if any(shared_contract(c)!=shared_contract(contracts[ARMS[0]]) for c in contracts.values()):
        raise ValueError('Arms differ in training budget or initialization policy')
    event('smoke_initialization',phase='load_common_parent')
    sampler,first_model,saved,bindings = build(contracts[ARMS[0]],device,checkpointing=True)
    parent = first_model.parent
    del sampler,saved
    generator = torch.Generator(device=device).manual_seed(contracts[ARMS[0]]['seed']+1229)
    x = torch.randn((1,8,16,16,16),device=device,generator=generator)
    t = torch.tensor([500.],device=device)
    initial_hash=parameter_count=parent_output_hash=None
    rows,input_sources = [],{}
    cache = {}
    for index,arm in enumerate(ARMS):
        model = first_model if index==0 else SharedEvidenceSS(parent,arm,
            seed=contracts[arm]['seed'],channels=contracts[arm]['residual_channels']).to(device)
        if index==0:
            first_model=None
        model.eval()
        versions = {name:p._version for name,p in parent.named_parameters()}
        init = tensor_hash(model.residual.state_dict())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != sum(p.numel() for p in model.residual.parameters()):
            raise ValueError('Unexpected trainable parent parameters')
        if initial_hash is None:
            initial_hash,parameter_count=init,count
        elif initial_hash!=init or parameter_count!=count:
            raise ValueError('New adapters differ in initialization or trainable capacity')
        _,sample,inputs,evidence,evidence_binding = load_case(obj,arm,cache)
        input_sources[arm]={'inputs':inputs,'evidence_binding':evidence_binding}
        context=sample['stock_condition'].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prepared=model.prepare(sample,device,evidence)
            with torch.no_grad():
                reference=model(x,t,context,prepared,enabled=False)
            prediction=model(x,t,context,prepared,enabled=True)
            if not torch.equal(reference,prediction):
                raise ValueError('Initial adapter changes parent forward: '+arm)
            current_parent_hash=tensor_binding(reference)
            if parent_output_hash is None:
                parent_output_hash=current_parent_hash
            elif current_parent_hash!=parent_output_hash:
                raise ValueError('Frozen parent output differs between observation arms')
            loss=prediction.float().square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite initialization objective')
        model.zero_grad(set_to_none=True)
        loss.backward()
        parent_clean=all(p.grad is None and p._version==versions[name] for name,p in parent.named_parameters())
        gradients=[p.grad for p in model.residual.projections.parameters()]
        nonzero=any(g is not None and bool(torch.any(g!=0)) for g in gradients)
        finite=all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        if not parent_clean or not nonzero or not finite:
            raise ValueError('Initial gradient isolation failed: '+arm)
        if tensor_hash(model.residual.state_dict())!=init:
            raise ValueError('Gradient check unexpectedly updated the new adapter')
        rows.append({'arm':arm,'contract':contracts[arm],'inputs':inputs,'evidence_binding':evidence_binding,
            'initial_residual_sha':init,'trainable_parameter_count':count,
            'parent_output':current_parent_hash,'new_output':tensor_binding(prediction),
            'zero_injection_exact':True,'parent_grad_none':True,'parent_versions_unchanged':True,
            'new_projection_gradient_nonzero':True,'new_projection_gradients_finite':True,
            'gradient_norm':float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients if g is not None))),
            'optimizer_steps':0,'objective':float(loss.detach())})
        model.zero_grad(set_to_none=True)
        del model,prepared,reference,prediction,loss,context,gradients
        release_cuda()
        event('smoke_initialization',phase='arm_complete',arm=arm)
    if input_sources[ARMS[0]]['inputs']!=input_sources[ARMS[1]]['inputs']:
        raise ValueError('Initialization used different image inputs')
    del parent,x,t,first_model
    release_cuda()
    write(path,{'time':now(),'identity':identity,'passed':True,
        'zero_initialization_exact':True,'matched_trainable_capacity':True,
        'parent_grad_none':True,'new_projection_gradient_nonzero':True,
        'initial_residual_sha':initial_hash,'trainable_parameter_count':parameter_count,
        'parent_checkpoint':contracts[ARMS[0]]['parent_checkpoint'],'parent_binding':bindings,
        'parent_loaded_once_shared_between_checks':True,'rows':rows,'input_sources':input_sources,
        'artifacts':[],'scope':'actual table2072 CUDA forwards and gradients; no optimizer update'})
    return binding(path)


def short_training_checks(identity,initial_binding):
    path=OUT/'smoke/training_check.json'
    previous=check_receipt(path,identity,('two_step_training_passed','shared_noise_order_exact',
        'matched_trainable_capacity','same_initial_residual'))
    if previous is not None:
        if previous['initial_check']!=initial_binding:
            raise ValueError('Short training was bound to a different initial gate')
        return binding(path)
    initial=read(verify(initial_binding))
    histories,runs,artifacts={},{},[initial_binding]
    for arm in ARMS:
        folder=OUT/'smoke'/('train_'+arm)
        event('smoke_two_step_training',phase='starting',arm=arm)
        train(arm,steps=STEPS,output=folder,max_steps=2)
        release_cuda()
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
            raise ValueError('Short checkpoint/run/partial contract mismatch: '+arm)
        if (run['initial_residual_sha']!=initial['initial_residual_sha']
                or run['trainable_parameter_count']!=initial['trainable_parameter_count']):
            raise ValueError('Training initialization/capacity differs from zero gate')
        if tensor_hash(checkpoint['residual'])==run['initial_residual_sha']:
            raise ValueError('Short training did not update the residual')
        history=[__import__('json').loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        if [r['step'] for r in history]!=[1,2]:
            raise ValueError('Short training history must contain exactly two steps')
        for row in history:
            if (row['arm']!=arm or any(not math.isfinite(row[key]) for key in
                    ('flow_loss','gradient_norm','diffusion_time','learning_rate'))
                    or row['gradient_norm']<=0 or len(row['noise_sha'])!=64):
                raise ValueError('Invalid short training loss/gradient/noise record')
        histories[arm]=history;runs[arm]=run
        artifacts += [binding(folder/name) for name in ('run.json','history.jsonl','last.pt','partial.json')]
        del checkpoint
        release_cuda()
        event('smoke_two_step_training',phase='complete',arm=arm)
    fields=('step','uid','index','original_index','epoch','offset','diffusion_time','noise_sha','learning_rate','object_group_exposures')
    paired=[]
    for left,right in zip(histories[ARMS[0]],histories[ARMS[1]]):
        if any(left[key]!=right[key] for key in fields):
            raise ValueError('Matched short training changed batch/time/noise/order')
        paired.append({key:left[key] for key in fields})
    for a,b in zip(runs[ARMS[0]]['data_identity'],runs[ARMS[1]]['data_identity']):
        if any(a[key]!=b[key] for key in ('uid','original_index','inputs','target_source','target_tensor_sha')):
            raise ValueError('Matched short training changed images or target')
    write(path,{'time':now(),'identity':identity,'passed':True,'initial_check':initial_binding,
        'two_step_training_passed':True,'shared_noise_order_exact':True,'matched_trainable_capacity':True,
        'same_initial_residual':True,'planned_formal_steps':STEPS,'short_steps':2,'paired_schedule':paired,
        'initial_residual_sha':initial['initial_residual_sha'],'trainable_parameter_count':initial['trainable_parameter_count'],
        'histories':histories,'artifacts':artifacts,'scope':'isolated 2-step runs; formal 200-step runs remain fresh'})
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
