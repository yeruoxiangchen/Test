import argparse
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F

from .config import BASE,OUT,MODELS,PARAMS,case,source,read,write,save_tensor,setup,lock_case,emit,now
from .sampling import masked_euler,join_original,restore_exterior
from .fields import lookup
from ...runtime import load_model,sparse,release
from ...geometry import observations,transform
from .evidence import grid_centers
from visual_evidence_probe.experiments.local_repair import photometric as photo


def sampler_configuration(stage):
    values=read(MODELS/'pipeline.json')['args'][stage+'_sampler']['params'].copy()
    steps=values.pop('steps');rescale=values.pop('rescale_t')
    return steps,rescale,values


def noise_like(value,seed):
    generator=torch.Generator(device=value.device).manual_seed(seed)
    return torch.randn(value.shape,device=value.device,dtype=value.dtype,generator=generator)


def ss_coordinates(decoder,latent):
    value=F.max_pool3d((decoder(latent)>0).float(),2,2)>.5
    return torch.argwhere(value)[:,[0,2,3,4]].int()


def compose_support(old,candidate,core,hull,positive,anchor=False):
    before=torch.zeros_like(core);o=old[:,1:].long();before[o[:,0],o[:,1],o[:,2]]=True
    c=candidate[:,1:].long();proposed=torch.zeros_like(core);proposed[c[:,0],c[:,1],c[:,2]]=True
    inside=proposed&core&(hull|before)
    if anchor:inside |=positive&core&hull
    combined=(before&~core)|inside
    cells=torch.argwhere(combined);coords=torch.cat((torch.zeros_like(cells[:,:1]),cells),1).int()
    requested=positive&core&hull
    receipt=dict(anchor=anchor,requested=int(requested.sum()),covered_before=int((requested&before).sum()),
                 covered_after=int((requested&combined).sum()),added=int((combined&~before).sum()),removed=int((before&~combined).sum()),
                 outside_support_exact=bool(torch.equal(combined[~core],before[~core])))
    if not receipt['outside_support_exact']:raise RuntimeError('SS exterior support differs')
    if anchor and receipt['covered_after']!=receipt['requested']:raise RuntimeError('verified SS anchor lost')
    return coords,receipt


def ss_guidance(decoder,latent,mask16,roi,matrix,index,baseline):
    if PARAMS['ss_guidance_steps']==0:return latent,[]
    _,k,w,_,masks=observations(index)
    observed=F.interpolate(torch.tensor(masks,device='cuda:0')[:,None].float(),size=(64,64),mode='area')[:,0]>.5
    original=F.interpolate(baseline['mask'][:,None].float(),size=(64,64),mode='area')[:,0]>.5
    deficit=observed&~original
    cells,centers=grid_centers(latent.device);world=transform(centers,matrix)
    hull=torch.tensor(roi['hull'],device=latent.device);envelope=torch.tensor(roi['envelope'],device=latent.device)
    positive=torch.tensor(roi['positive'],device=latent.device)
    negative=~hull&envelope
    projections=[]
    for view in roi['train']:
        uv,z=photo.project(world[:,None],k[view],w[view]);uv=uv[:,0];z=z[:,0]
        pixel=torch.floor((uv+.5)*64/518).long()
        valid=(pixel>=0).all(1)&(pixel<64).all(1)&(z>0)&hull.flatten()&envelope.flatten()
        index2=(pixel[:,1].clamp(0,63)*64+pixel[:,0].clamp(0,63))
        projections.append((view,index2,valid))
    delta=torch.nn.Parameter(torch.zeros_like(latent));optimizer=torch.optim.Adam([delta],lr=PARAMS['ss_guidance_lr'])
    curve=[]
    for step in range(PARAMS['ss_guidance_steps']):
        optimizer.zero_grad(set_to_none=True)
        candidate=latent+PARAMS['ss_guidance_cap']*torch.tanh(delta)*mask16
        logits=F.max_pool3d(decoder(candidate),2,2)[0,0]
        loss=logits.sum()*0
        if bool(positive.any()):loss=loss+F.softplus(-logits[positive]).mean()
        if bool(negative.any()):loss=loss+.2*F.softplus(logits[negative]).mean()
        probability=logits.sigmoid().flatten()
        for view,pixels,valid in projections:
            if not bool(deficit[view].any()):continue
            coverage=probability.new_zeros(64*64).scatter_reduce(0,pixels,probability*valid,reduce='amax',include_self=True)
            coverage=F.max_pool2d(coverage.reshape(1,1,64,64),3,1,1)[0,0]
            loss=loss+F.relu(.65-coverage)[deficit[view]].mean()
        loss=loss+.001*torch.tanh(delta).square().mean()
        (loss*512).backward();delta.grad.div_(512)
        if not bool(torch.isfinite(delta.grad).all()):raise RuntimeError('invalid SS occupancy gradient')
        torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step();curve.append(float(loss.detach()))
    result=(latent+PARAMS['ss_guidance_cap']*torch.tanh(delta)*mask16).detach()
    if not torch.equal(result[~mask16.expand_as(result)],latent[~mask16.expand_as(latent)]):raise RuntimeError('SS exterior latent changed')
    return result,curve


def regenerate_ss(index,seed,force=False):
    root=case(index,seed);path=root/'ss_complete.json'
    if path.exists():return
    started=time.monotonic()
    plan=torch.load(root/'plan.pt',weights_only=False);roi=plan['roi'];roi['train']=read(source(index,seed)/'evidence.json')['train_views']
    if not roi['has_evidence'] or (not roi['needs_ss'] and not force):
        write(path,dict(time=now(),variants=[],skipped='no reliable support-update request',gt_used=False));return
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    sampler=FlowEulerGuidanceIntervalSampler(PARAMS['sigma_min']);steps,rescale,kwargs=sampler_configuration('sparse_structure')
    context=torch.load(BASE/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda()
    kwargs['neg_cond']=torch.zeros_like(context)
    flow=load_model('ss_flow',binding_root=OUT);decoder=load_model('ss_decoder',binding_root=OUT)
    old=torch.load(source(index,seed)/'ss.pt',weights_only=False)['coords'].cuda()
    replay_path=root/'ss_dense_replay.pt'
    if replay_path.exists():clean=torch.load(replay_path,weights_only=False)['latent'].cuda()
    else:
        torch.manual_seed(seed);noise=torch.randn((1,8,16,16,16),device='cuda:0')
        with torch.no_grad():clean=sampler.sample(flow,noise,cond=context,steps=steps,rescale_t=rescale,verbose=False,**kwargs).samples
        with torch.no_grad():replayed=ss_coordinates(decoder,clean)
        if not torch.equal(replayed,old):raise RuntimeError('official SS replay differs from original support')
        save_tensor(replay_path,dict(latent=clean.cpu(),coords=replayed.cpu(),native_replay_exact=True))
    mask32=torch.tensor(roi['envelope'],device='cuda:0')[None,None].float()
    mask16=F.max_pool3d(mask32,2,2).bool()
    core=torch.tensor(roi['core'],device='cuda:0');hull=torch.tensor(roi['hull'],device='cuda:0')
    oc=old[:,1:].long();old_grid=torch.zeros_like(core);old_grid[oc[:,0],oc[:,1],oc[:,2]]=True
    baseline={key:value.cuda() for key,value in torch.load(root/'registered_render.pt',weights_only=False).items()}
    matrix=plan['matrix'].cuda();variants=[]
    for start,proposal_seed in zip(PARAMS['ss_times'],PARAMS['proposal_seeds']):
        name=f'ss_t{int(start*100):03d}_n{proposal_seed}'
        with torch.no_grad():
            sampled=masked_euler(sampler,flow,clean,mask16,noise_like(clean,index*100000+seed*1000+proposal_seed),
                      start,steps,rescale,context,kwargs=kwargs)
        sampled,curve=ss_guidance(decoder,sampled,mask16,roi,matrix,index,baseline)
        with torch.no_grad():candidate=ss_coordinates(decoder,sampled)
        combined,coverage=compose_support(old,candidate,core,hull,torch.tensor(roi['positive'],device='cuda:0'),PARAMS.get('anchor_support',False))
        if not len(combined):raise RuntimeError('empty regenerated support')
        # Outside the core, support is exactly the original coordinate set.
        save_tensor(root/f'{name}.pt',dict(coords=combined.cpu(),latent=sampled.cpu(),guidance_curve=curve,
                    exterior_support_exact=True,gt_used=False,coverage=coverage,
                    support_source='native local SS prior plus RGB-verified anchors' if PARAMS.get('anchor_support',False) else 'native local SS prior'))
        variants.append(dict(name=name,path=str(root/f'{name}.pt'),tokens=len(combined),start=start,coverage=coverage))
        emit(index,seed,'ss_variant',name=name,tokens=len(combined))
    write(path,dict(time=now(),variants=variants,native_replay_exact=True,gt_used=False,
                   diagnostic_only=bool(force and not roi['needs_ss']),seconds=time.monotonic()-started))
    del flow,decoder;release()


def regenerate_shape(index,seed):
    root=case(index,seed);path=root/'sampling_complete.json'
    if path.exists():return
    started=time.monotonic()
    plan=torch.load(root/'plan.pt',weights_only=False);roi=plan['roi']
    if not roi['has_evidence']:
        write(path,dict(time=now(),candidates=[],gt_used=False,reason='no confirmed edit region'));return
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    sampler=FlowEulerGuidanceIntervalSampler(PARAMS['sigma_min']);steps,rescale,kwargs=sampler_configuration('shape_slat')
    context=torch.load(BASE/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda();kwargs['neg_cond']=torch.zeros_like(context)
    flow=load_model('shape_flow',binding_root=OUT)
    old=torch.load(source(index,seed)/'original_latent.pt',weights_only=False)
    original_features=old['features'].cuda().float();original_coords=old['coords'].cuda();mean=old['mean'].cuda();std=old['std'].cuda()
    envelope=torch.tensor(roi['envelope'],device='cuda:0');candidates=[]
    configs=[dict(arm='shape',name=f'shape_t{int(t*100):03d}_n{n}',start=t,noise=n,coords=original_coords)
             for t in PARAMS['shape_times'] for n in PARAMS['proposal_seeds']]
    if (root/'ss_complete.json').exists() and roi['needs_ss']:
        for variant in read(root/'ss_complete.json')['variants']:
            coords=torch.load(variant['path'],weights_only=False)['coords'].cuda()
            configs.extend(dict(arm='ss_shape',name=f'{variant["name"]}_shape_n{n}',start=1.,noise=n,coords=coords)
                           for n in PARAMS['proposal_seeds'])
    for config in configs:
        dest=root/'candidates'/f'{config["name"]}.pt'
        if dest.exists():
            candidates.append({key:value for key,value in config.items() if key!='coords'}|dict(path=str(dest)));continue
        coords=config['coords'];index_old,present=join_original(original_coords,coords)
        clean=torch.zeros((len(coords),32),device='cuda:0')
        clean[present]=(original_features[index_old[present]]-mean)/std
        editable=lookup(envelope,coords[:,1:]).bool()
        if bool((~present&~editable).any()):raise RuntimeError('new coordinates outside edit envelope')
        _,retained=join_original(coords,original_coords)
        old_outside=~lookup(envelope,original_coords[:,1:]).bool()
        if not bool(retained[old_outside].all()):raise RuntimeError('original exterior SS cell lost')
        if config['start']<1 and not bool(present.all()):raise RuntimeError('new SS cells require full-noise generation')
        noise=noise_like(clean,index*100000+seed*1000+config['noise'])
        with torch.no_grad():
            sampled=masked_euler(sampler,flow,clean,editable[:,None],noise,config['start'],steps,rescale,context,
                     wrap=lambda x:sparse(x,coords),unwrap=lambda x:x.feats,kwargs=kwargs)
            features=restore_exterior(sampled*std+mean,original_features,original_coords,coords,editable)
        save_tensor(dest,dict(features=features.cpu(),coords=coords.cpu(),mean=mean.cpu(),std=std.cpu(),
                space=old['space'],resolution=512,exterior_features_exact=True,gt_used=False,
                start=config['start'],noise_seed=config['noise'],arm=config['arm']))
        candidates.append({key:value for key,value in config.items() if key!='coords'}|dict(path=str(dest)))
        emit(index,seed,'shape_variant',name=config['name'],tokens=len(coords))
    write(path,dict(time=now(),candidates=candidates,gt_used=False,weights_frozen=True,seconds=time.monotonic()-started))
    del flow;release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--stage',choices=('ss','shape'),required=True);p.add_argument('--force',action='store_true');a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.7)
    with lock_case(a.index,a.seed):
        if a.stage=='ss':regenerate_ss(a.index,a.seed,a.force)
        else:regenerate_shape(a.index,a.seed)


if __name__=='__main__':main()
