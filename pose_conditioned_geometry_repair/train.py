"""Adapter/LoRA flow matching with damaged-state conditioning and strict resume."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import datetime
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import common as c
from .data import RepairDataset, intervention
from .model import build, conditional_sample, features, replace
from .flow import diffuse,target_velocity,weighted_flow_loss,bridge_state
from .vendor_pose.conditioning import build_2d_context, select_view_indices
from .vendor_pose.checkpoint import (initialize_ema, update_ema, ema_ramp_decay,
    save_checkpoint, load_checkpoint, update_checkpoint_alias, rng_state, restore_rng_state)
from .vendor_pose.models import optimizer_parameter_groups, trainable_named_parameters


class Forward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, context, sample, views,stock=None):
        return self.model.adapted_prediction(x,t,context,sample,view_indices=views,
            stock_velocity=stock,compare_stock=False)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['ss','shape'], required=True)
    p.add_argument('--arm', choices=['repair','full'], default='repair')
    p.add_argument('--subset', choices=['overfit','train'], default='overfit')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--save-every', type=int, default=200)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--new-lr', type=float, default=1e-4)
    p.add_argument('--lora-lr', type=float, default=3e-5)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--resume', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--objective',choices=['flow','bridge'],default='flow')
    return p


def main():
    args = parser().parse_args()
    if args.objective=='bridge' and args.arm!='repair':
        raise ValueError('The full reconstruction control uses the standard flow objective')
    if min(args.steps,args.grad_accum,args.save_every,args.log_every) < 1:
        raise ValueError('Positive update counts required')
    rank = int(os.environ.get('RANK',0))
    local = int(os.environ.get('LOCAL_RANK',0))
    world = int(os.environ.get('WORLD_SIZE',1))
    if world > 1:
        dist.init_process_group('nccl',timeout=datetime.timedelta(hours=12))
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed+rank*100003)
    random.seed(args.seed+rank*100003)
    np.random.seed(args.seed+rank*100003)
    ds = RepairDataset(c.DATA/f'{args.subset}.json',args.stage,cache_size=32 if args.subset=='overfit' else 4)
    out = Path(args.output).resolve()
    if not out.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Training output must be inside the task run')
    model = build(args.stage,args.arm,device)
    model.train()
    named = trainable_named_parameters(model)
    surface = {'trainable_parameters':sum(p.numel() for p in named.values()),
               'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
               'lora_tensors':sum('lora_' in n for n in named),
               'damaged_state_parameters':sum(p.numel() for n,p in named.items() if n.startswith('damaged_condition.'))}
    identity = {'stage':args.stage,'arm':args.arm,'base_model':c.MODEL_PATHS[args.stage+'_flow'],
                'objective':args.objective,
                'base_initialization':'official_trellis2', 'old_30k_loaded':False,
                'distillation':False,'surface':surface,
                'coordinate_contract':'native T2, [-.5,.5]^3; OpenCV +z forward, +y down',
                'shape_support_training':'native GT support; predicted SS rollout evaluated separately',
                'image_context':'first selected DINOv3 full 1029 tokens',
                'geometry_condition':'damaged features, editable ROI, present flag, canonical xyz',
                'source':c.source_identity()['identity'],
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu_name':torch.cuda.get_device_name(device)}
    data_identity = ds.identity()
    data_identity['plan_identity'] = c.plan()['identity']
    data_identity['effective_selection_identity']=c.read(c.DATA/f'{args.subset}.json')['plan_identity']
    opt = torch.optim.AdamW(optimizer_parameter_groups(model,new_lr=args.new_lr,
        lora_lr=args.lora_lr,new_weight_decay=.01),betas=(.9,.95))
    for g in opt.param_groups:
        g['base_lr']=g['lr']
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    ema = initialize_ema(model)
    step=micro=0
    history=[]
    if args.resume:
        ck=load_checkpoint(args.resume,model=model,stage=args.stage,weights='raw')
        if ck['data_identity'] != data_identity:
            legacy=dict(data_identity)
            legacy.pop('effective_selection_identity',None)
            if not (args.subset=='overfit' and ck['data_identity']==legacy):
                raise ValueError('Resume dataset differs')
        for key in ['stage','arm','base_model','geometry_condition']:
            if ck['model_identity'][key]!=identity[key]:
                raise ValueError('Resume model differs: '+key)
        for key in ['stage','arm','subset','grad_accum','seed','new_lr','lora_lr','warmup']:
            if ck['args'][key]!=vars(args)[key]:
                raise ValueError('Resume training protocol differs: '+key)
        if ck['args'].get('objective','flow')!=args.objective:
            raise ValueError('Resume objective differs')
        opt.load_state_dict(ck['optimizer'])
        ema={n:v.to(device) for n,v in ck['ema_trainable_state'].items()}
        step,micro=int(ck['step']),int(ck['micro_step'])
        history=ck['history']
        if len(ck['rng_by_rank'])!=world:
            raise ValueError('Resume world size differs')
        restore_rng_state(ck['rng_by_rank'][rank])
    wrapped = Forward(model)
    if world>1:
        wrapped=DDP(wrapped,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    if rank==0:
        c.write(out/'config.json',{'args':vars(args),'model':identity,'data':data_identity,'time':c.now()})
        print(json.dumps({'initialized':True,**surface,'objects':len(ds),'world_size':world}),flush=True)
    opt.zero_grad(set_to_none=True)
    start=time.time()
    accumulation=[]
    audit_done=bool(args.resume)
    gradient_audit=None
    while step<args.steps:
        # Epoch permutations depend only on epoch/seed, not process RNG. A
        # resumed run can reconstruct its exact object position.
        position=micro*world+rank
        epoch=position//len(ds)
        order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(args.seed+epoch*1000003))
        idx=int(order[position%len(ds)])
        sample=ds[idx]
        variant=(0 if args.subset=='overfit' else int(torch.randint(0,4,(),
            generator=torch.Generator().manual_seed(args.seed+position*1000003+1703)).item()))
        pair=intervention(sample,args.stage,variant,device)
        clean=pair['clean']
        noise=replace(clean,torch.randn_like(features(clean)))
        if args.objective=='bridge':
            t=1. if float(torch.rand(()))<.5 else float(torch.rand(()))
            x=bridge_state(clean,pair['bad'],noise,t)
            velocity=features(pair['bad'])-features(clean)
        else:
            t=float(.05+.85*torch.rand(()).item())
            x=diffuse(clean,noise,t)
            velocity=target_velocity(clean,noise)
        views=(torch.arange(len(sample['dino_features'])) if args.subset=='overfit' else
               select_view_indices(len(sample['dino_features']),minimum=3,maximum=8,
                   generator=torch.Generator().manual_seed(args.seed+position*1000003+2701)))
        context=build_2d_context(sample['dino_features'],views).to(device)
        image_enabled=args.subset=='overfit' or random.random()>=.1
        if not image_enabled:
            context=torch.zeros_like(context)
        cond=conditional_sample(sample,pair,image_enabled)
        timestep=torch.tensor([(650 if args.objective=='bridge' else 1000)*t],device=device)
        if not audit_done:
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                stock=model.stock_prediction(x,timestep,context)
                pred,_=model.adapted_prediction(x,timestep,context,cond,view_indices=views,stock_velocity=stock,compare_stock=False)
            err=float((features(pred).float()-features(stock).float()).abs().max())
            if err!=0:
                raise ValueError('Fresh adaptation changed stock prediction')
            if rank==0:
                c.write(out/'initial_stock_audit.json',{'max_abs':err,'passed':True,'time':c.now()})
            del stock,pred
            model.train()
            audit_done=True
        synchronized=(micro+1)%args.grad_accum==0
        bridge_stock=None
        if args.objective=='bridge':
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                bridge_stock=model.stock_prediction(x,timestep,context)
        sync=wrapped.no_sync() if world>1 and not synchronized else nullcontext()
        with sync:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,stats=wrapped(x,timestep,context,cond,views,bridge_stock)
                if bridge_stock is not None:
                    pred=replace(pred,features(pred).float()-features(bridge_stock).float())
                loss,inside,outside=weighted_flow_loss(pred,velocity,pair['mask'],args.stage,args.arm)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite flow loss')
            (loss/args.grad_accum).backward()
        micro+=1
        accumulation.append((float(loss.detach()),float(inside.detach()),float(outside.detach())))
        if not synchronized:
            continue
        parameters=list(named.values())
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Nonfinite trainable gradients')
        # Only the frozen official backbone is allowed to be gradient-free.
        if step<3:
            group_norms={}
            for prefix in ['aggregator.','block_condition.','damaged_condition.','flow.']:
                group_norms[prefix]=float(sum(p.grad.detach().float().square().sum() for n,p in named.items()
                    if n.startswith(prefix) and p.grad is not None).sqrt().item())
            if step==0 and group_norms['block_condition.']<=0:
                raise ValueError('Zero projection gradients are disconnected')
            if step==2 and args.arm=='repair' and group_norms['damaged_condition.']<=0:
                raise ValueError('Damaged-state input is disconnected')
            gradient_audit={'time':c.now(),'update':step+1,'norms':group_norms,
                'frozen_backbone_has_grad':any(p.grad is not None for p in model.parameters() if not p.requires_grad)}
            if gradient_audit['frozen_backbone_has_grad']:
                raise ValueError('Frozen backbone received a gradient')
            if rank==0:
                c.write(out/'gradient_audit.json',gradient_audit)
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
        factor=min(1.,(step+1)/max(1,args.warmup))
        for g in opt.param_groups:
            g['lr']=g['base_lr']*factor
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not all(torch.isfinite(p).all() for p in parameters):
            raise ValueError('Optimizer produced nonfinite weights')
        step+=1
        update_ema(model,ema,decay=ema_ramp_decay(.9995,step))
        values=torch.tensor(np.mean(accumulation,axis=0),device=device)
        if world>1:
            dist.all_reduce(values)
            values/=world
        accumulation.clear()
        rec={'time':c.now(),'step':step,'micro_step':micro,'flow_loss':float(values[0]),
             'edit_flow_loss':float(values[1]),'outside_flow_loss':float(values[2]),
             'gradient_norm':norm,'elapsed_seconds':time.time()-start,
             'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'objects':len(ds)}
        if rank==0 and (step==1 or step%args.log_every==0 or step==args.steps):
            history.append(rec)
            print(json.dumps(rec),flush=True)
            c.write(out/'progress.json',rec)
        if step%args.save_every==0 or step==args.steps:
            rngs=[rng_state()]
            if world>1:
                rngs=[None]*world
                dist.all_gather_object(rngs,rng_state())
            if rank==0:
                ckpath=out/'checkpoints'/f'step_{step:06d}.pt'
                save_checkpoint(ckpath,model=model,optimizer=opt,scaler=scaler,ema=ema,
                    stage=args.stage,step=step,micro_step=micro,args=vars(args),data_identity=data_identity,
                    model_identity=identity,history=history,rng_by_rank=rngs)
                update_checkpoint_alias(ckpath,out/'checkpoints/last.pt')
                c.write(out/'checkpoint_receipt.json',{'time':c.now(),'step':step,
                    'path':str(ckpath),'data_identity':data_identity,'source_identity':identity['source']})
            if world>1:
                dist.barrier()
    if rank==0:
        c.write(out/'complete.json',{'time':c.now(),'step':step,'success':True,
            'elapsed_seconds':time.time()-start,'model':identity,'data':data_identity})
    if world>1:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
