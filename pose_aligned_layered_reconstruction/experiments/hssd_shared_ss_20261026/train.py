"""Paired shared-input residual training, with strict and resumable cursors."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ...frozen_slat import configure_determinism
from ..hssd_cell_query_20261023.common import tensor_hash
from ..hssd_cell_query_20261023.train import fixed_batch, fixed_noise, recover_history
from .common import ARMS, OUT, STEPS, FORMAT, policy, plan, verify, write, now, binding
from .data import load_case
from .model import build

def atomic_save(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def train(arm,steps=STEPS,output=None,max_steps=None):
    configure_determinism()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    contract = policy(arm,steps=steps)
    steps = int(contract['steps'])
    if max_steps is not None and (max_steps<1 or max_steps>steps):
        raise ValueError('max_steps must be within the fixed training budget')
    stop_step = steps if max_steps is None else int(max_steps)
    output = Path(output) if output else OUT/'train'/arm
    output.mkdir(parents=True,exist_ok=True)
    with (output/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_path,last = output/'run.json',output/'last.pt'
        if run_path.exists() and json.loads(run_path.read_text())['contract']!=contract:
            raise ValueError('Training policy changed; use an isolated experiment')
        checkpoint = torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
        if checkpoint and not run_path.exists():
            raise ValueError('Checkpoint exists without its training run metadata')
        if checkpoint and (checkpoint['format']!=FORMAT or checkpoint['contract']!=contract):
            raise ValueError('Checkpoint contract differs')
        if checkpoint and not 0<=checkpoint['step']<=steps:
            raise ValueError('Checkpoint cursor is outside fixed training budget')
        objects = plan()['plan']['objects']
        if len(objects)!=4 or [o['original_index'] for o in objects]!=[0,5,6,7]:
            raise ValueError('Require all four original shared-scene objects')
        cache,cases,data_identity = {},[],[]
        for obj in objects:
            source,sample,inputs,evidence,evidence_binding = load_case(obj,arm,cache)
            if source['object_uid']!=obj['uid']:
                raise ValueError('Training target/input object differs')
            if (source['target'].shape!=(8,16,16,16) or source['target'].dtype!=torch.float32
                    or not torch.isfinite(source['target']).all()):
                raise ValueError('Invalid archived SS latent target')
            cases.append((source,sample,evidence))
            data_identity.append({'uid':obj['uid'],'original_index':obj['original_index'],
                'inputs':inputs,'evidence_binding':evidence_binding,
                'target_source':obj['ss'],'target_tensor_sha':tensor_hash({'target':source['target']})})
        if checkpoint and checkpoint['data_identity']!=data_identity:
            raise ValueError('Resume data identity differs')
        start_step = checkpoint['step'] if checkpoint else 0
        history = output/'history.jsonl'
        recover_history(history,start_step)
        if start_step>stop_step:
            raise ValueError('Requested max_steps is behind completed checkpoint')
        if start_step==steps:
            run = json.loads(run_path.read_text())
            if (run['bindings']!=checkpoint['bindings']
                    or run['initial_residual_sha']!=checkpoint['initial_residual_sha']
                    or run['data_identity']!=data_identity
                    or checkpoint.get('frozen_parameter_versions_unchanged') is not True):
                raise ValueError('Completed checkpoint/run metadata differs')
            result = {'step':steps,'contract':contract,
                'final_residual_sha':tensor_hash(checkpoint['residual']),
                'data_identity':data_identity,'checkpoint':binding(last)}
            complete = output/'complete.json'
            if complete.exists():
                old = json.loads(complete.read_text())
                if any(old.get(k)!=v for k,v in result.items()):
                    raise ValueError('Completed training record changed')
                verify(old['checkpoint'])
            else:
                write(complete,{'time':now(),**result})
            return
        if start_step==stop_step:
            write(output/'partial.json',{'time':now(),'step':start_step,'planned_steps':steps,
                'contract':contract,'checkpoint':binding(last),'reason':'requested bounded short run'})
            return
        sampler,model,saved,bindings = build(contract,device,checkpointing=True)
        model.train()
        versions = {name:p._version for name,p in model.parent.named_parameters()}
        params = list(model.residual.parameters())
        trainable_names = [name for name,p in model.named_parameters() if p.requires_grad]
        if (any(p.requires_grad for p in model.parent.parameters())
                or any(not name.startswith('residual.') for name in trainable_names)):
            raise ValueError('Only the new matched residual may train')
        optimizer = torch.optim.AdamW(params,lr=contract['lr'],betas=tuple(contract['betas']),
            weight_decay=contract['weight_decay'])
        init_hash = tensor_hash(model.residual.state_dict())
        if checkpoint:
            if checkpoint['bindings']!=bindings or checkpoint['initial_residual_sha']!=init_hash:
                raise ValueError('Frozen parent or new-adapter initialization changed')
            model.residual.load_state_dict(checkpoint['residual'],strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
        run = {'contract':contract,'data_identity':data_identity,'bindings':bindings,
            'initial_residual_sha':init_hash,'trainable_parameter_count':sum(p.numel() for p in params),
            'trainable_parameter_names':trainable_names,'format':FORMAT,
            'fixed_parent':'complete old occluded appearance_control including its trained residual'}
        if run_path.exists() and json.loads(run_path.read_text())!=run:
            raise ValueError('Training metadata differs on resume')
        write(run_path,run)
        prepared_cache = {}
        prior_elapsed = 0.
        if history.exists() and history.stat().st_size:
            prior_elapsed = json.loads(history.read_text().splitlines()[-1]).get('elapsed_seconds',0.)
        started = time.time()
        for step in range(start_step,stop_step):
            index,epoch,offset = fixed_batch(objects,contract['seed'],step)
            source,sample,evidence = cases[index]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if index not in prepared_cache:
                    prepared_cache[index] = {k:v.cpu() for k,v in model.prepare(sample,device,evidence).items()}
                prepared = {k:v.to(device) for k,v in prepared_cache[index].items()}
                target = source['target'].to(device).float()[None]
                t,noise = fixed_noise(target,contract['seed'],step)
                noise_sha = tensor_hash({'noise':noise})
                xt,velocity = sampler._get_model_gt(target,t,noise)
                prediction = model(xt,target.new_tensor([t*1000]),sample['stock_condition'].to(device),prepared)
                loss = F.mse_loss(prediction.float(),velocity.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            lr = contract['lr']*min(1.,(step+1)/contract['warmup'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if any(p._version!=versions[name] or p.grad is not None for name,p in model.parent.named_parameters()):
                raise RuntimeError('Frozen parent changed or received gradients')
            row = {'time':now(),'step':step+1,'arm':arm,'uid':objects[index]['uid'],
                'index':index,'original_index':objects[index]['original_index'],'epoch':epoch,'offset':offset,
                'flow_loss':float(loss.detach()),'gradient_norm':float(norm),'diffusion_time':t,
                'noise_sha':noise_sha,'learning_rate':lr,'object_group_exposures':step+1,
                'elapsed_seconds':prior_elapsed+time.time()-started,
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
            if step==start_step or (step+1)%20==0:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    row.update(model.metrics(prepared))
            with history.open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
            if (step+1)%20==0 or step+1==stop_step:
                atomic_save(last,{'format':FORMAT,'step':step+1,'contract':contract,
                    'data_identity':data_identity,'bindings':bindings,'initial_residual_sha':init_hash,
                    'residual':{name:v.detach().cpu().clone() for name,v in model.residual.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'frozen_parameter_versions_unchanged':True})
        if stop_step==steps:
            write(output/'complete.json',{'time':now(),'step':steps,'contract':contract,
                'final_residual_sha':tensor_hash(model.residual.state_dict()),
                'data_identity':data_identity,'checkpoint':binding(last)})
        else:
            write(output/'partial.json',{'time':now(),'step':stop_step,'planned_steps':steps,
                'contract':contract,'checkpoint':binding(last),'reason':'requested bounded short run'})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--output')
    parser.add_argument('--max-steps',type=int)
    args = parser.parse_args()
    train(args.arm,args.steps,args.output,args.max_steps)
