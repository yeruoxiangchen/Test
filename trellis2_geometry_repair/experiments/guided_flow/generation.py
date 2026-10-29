import argparse
import time
import torch

from .config import OUT,BASE,PARAMS,DIAGNOSTIC,case,previous,original,save,read,write,now,setup,lock_case,emit
from .backend import decode,protection
from .data import load,record,cpu,device,prepare_diagnostics
from .guide import Guide
from .integration import integrate
from ...runtime import load_model,release
from ..local_regeneration.fields import lookup
from ..local_regeneration.sampling import join_original,restore_exterior
from ..local_regeneration.generation import sampler_configuration,noise_like

def one(index,seed,mode,context):
    root=case(index,seed);receipt=root/f'generation_{mode}.json'
    if receipt.exists():return
    started=time.monotonic();data=load(index,seed,context)
    if mode=='diagnostic':prepare_diagnostics(index,seed,data,context)
    old=torch.load(original(index,seed)/'original_latent.pt',weights_only=False)
    original_features=old['features'].cuda();original_coords=old['coords'].cuda();mean=old['mean'].cuda();std=old['std'].cuda()
    coords=torch.load(previous(index,seed)/(PARAMS['support']+'.pt'),weights_only=False)['coords'].cuda()
    index_old,present=join_original(original_coords,coords)
    clean=torch.zeros((len(coords),32),device='cuda:0');clean[present]=(original_features[index_old[present]]-mean)/std
    active=lookup(data['envelope'],coords[:,1:]).bool()
    if bool((~present&~active).any()):raise RuntimeError('new coarse support outside envelope')
    decoder=load_model('shape_decoder',checkpoint=True,binding_root=OUT)
    refpath=root/'field_reference.pt'
    if refpath.exists():reference=device(torch.load(refpath,weights_only=False))
    else:
        with torch.no_grad():reference=protection(decode(decoder,original_features,original_coords),data['envelope'],PARAMS['field_anchors'],PARAMS['subdivision_anchors'])
        save(refpath,cpu(reference))
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    sampler=FlowEulerGuidanceIntervalSampler(PARAMS['sigma_min']);steps,rescale,kwargs=sampler_configuration('shape_slat')
    condition=torch.load(BASE/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda();kwargs['neg_cond']=torch.zeros_like(condition)
    flow=load_model('shape_flow',binding_root=OUT)
    configs=([('prior',n) for n in PARAMS['bank_seeds']]+[('input',n) for n in PARAMS['proposal_seeds']]) if mode=='method' else [(arm,n) for arm in DIAGNOSTIC for n in PARAMS['proposal_seeds']]
    specs=[]
    for arm,n in configs:
        name=f'{arm}_n{n}';path=root/'candidates'/f'{name}.pt';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():specs.append(read(meta));continue
        rec,domain=record(data,reference,arm,index,seed)
        guider=Guide(decoder,flow,coords,mean,std,rec,context) if arm!='prior' else None
        noise=noise_like(clean,index*100000+seed*1000+n)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        sampled=integrate(sampler,flow,clean,active[:,None],noise,condition,dict(kwargs,_coords=coords),
                          guide=guider,steps=steps,rescale=rescale,guide_steps=PARAMS['guide_steps'])
        torch.cuda.synchronize();seconds=time.monotonic()-begin
        features=restore_exterior(sampled*std+mean,original_features,original_coords,coords,active)
        if arm=='prior' and n in PARAMS['proposal_seeds']:
            expected=torch.load(previous(index,seed)/'candidates'/f'ss_t065_n101_shape_n{n}.pt',weights_only=False)
            if not torch.equal(features.cpu(),expected['features']):raise RuntimeError('unguided replay differs from frozen V2')
        spec=dict(name=name,arm=arm,noise=n,path=str(path),seconds=seconds,flow_calls=steps,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30,domain=domain,
                  guidance=guider.curve if guider else [],GT_used=arm in DIAGNOSTIC)
        save(path,dict(features=features.cpu(),coords=coords.cpu(),mean=old['mean'],std=old['std'],
             resolution=512,space=old['space'],exterior_features_exact=True,SS_fixed=True,GT_used=arm in DIAGNOSTIC))
        write(meta,spec);specs.append(spec)
        write(root/f'generation_{mode}_progress.json',dict(time=now(),candidates=specs))
        emit(index,seed,'sampled',arm=arm,noise=n,seconds=round(seconds,2),guidance_accepted=sum(r.get('accepted',False) for r in spec['guidance']))
        del sampled,features,rec,guider;release()
    write(receipt,dict(time=now(),candidates=specs,seconds=time.monotonic()-started,GT_used=mode=='diagnostic',
                      same_SS_and_noises=True,weights_frozen=True,common_initialization_in_total_only=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=('method','diagnostic'),required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed):one(a.index,a.seed,a.mode,dr.RasterizeCudaContext(device='cuda:0'))

if __name__=='__main__':main()
