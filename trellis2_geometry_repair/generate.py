import argparse
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import OUT,MODELS,PARAMS,folder,read,write,setup,snapshot,emit,now
from .runtime import load_model,release,sparse,normalization,decode_native,save_mesh


def encode(indices):
    from .features import compatible_extractor
    extractor=None
    for index in indices:
        root=OUT/'data'/f'{index:03d}';path=root/'reference_context.pt'
        if path.exists():continue
        if extractor is None:
            extractor=compatible_extractor(MODELS/'dinov3');extractor.to('cuda:0')
        rec=read(root/'observations.json');image=Image.open(rec['images'][PARAMS['reference_view']]['path']).convert('RGB')
        features=extractor([image])
        if tuple(features.shape)!=(1,1029,1024):raise RuntimeError(f'wrong native image context {features.shape}')
        torch.save(dict(features=features.float().cpu(),view=PARAMS['reference_view'],image=rec['images'][0]['path']),path)
    del extractor;release()


def sample_ss(indices,seeds):
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    pending=[(i,s) for i in indices for s in seeds if not (folder(i,s)/'ss.pt').exists()]
    if not pending:return
    flow=load_model('ss_flow');decoder=load_model('ss_decoder')
    params=read(MODELS/'pipeline.json')['args']['sparse_structure_sampler']['params'];sampler=FlowEulerGuidanceIntervalSampler(1.e-5)
    with torch.no_grad():
        for index,seed in pending:
            torch.manual_seed(seed);start=time.monotonic()
            context=torch.load(OUT/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda()
            noise=torch.randn((1,8,16,16,16),device='cuda:0')
            result=sampler.sample(flow,noise,cond=context,neg_cond=torch.zeros_like(context),verbose=False,**params)
            decoded=decoder(result.samples)>0
            decoded=F.max_pool3d(decoded.float(),2,2)>0.5
            coords=torch.argwhere(decoded)[:,[0,2,3,4]].int()
            if not len(coords):raise RuntimeError('native SS generated empty support')
            torch.save(dict(coords=coords.cpu(),cuda_rng=torch.cuda.get_rng_state().cpu(),
                            sampler=params,seconds=time.monotonic()-start),folder(index,seed)/'ss.pt')
            emit(index,seed,'ss_complete',tokens=len(coords))
            del result,decoded,context,noise
    del flow,decoder;release()


def sample_shape(indices,seeds):
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    pending=[(i,s) for i in indices for s in seeds if not (folder(i,s)/'shape_complete.json').exists()]
    if not pending:return
    flow=load_model('shape_flow');params=read(MODELS/'pipeline.json')['args']['shape_slat_sampler']['params']
    sampler=FlowEulerGuidanceIntervalSampler(1.e-5);mean,std=normalization('cuda:0')
    with torch.no_grad():
        for index,seed in pending:
            root=folder(index,seed);start=time.monotonic();ss=torch.load(root/'ss.pt',weights_only=False)
            coords=ss['coords'].cuda();torch.cuda.set_rng_state(ss['cuda_rng'])
            context=torch.load(OUT/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda()
            noise=sparse(torch.randn((len(coords),32),device='cuda:0'),coords)
            result=sampler.sample(flow,noise,cond=context,neg_cond=torch.zeros_like(context),verbose=False,**params)
            features=result.samples.feats.float()*std+mean
            if not bool(torch.isfinite(features).all()):raise RuntimeError('nonfinite Shape features')
            torch.save(dict(features=features.cpu(),coords=coords.cpu(),mean=mean.cpu(),std=std.cpu(),
                            space='native denormalized decoder features',resolution=512),root/'original_latent.pt')
            write(root/'shape_complete.json',dict(time=now(),sampler=params,tokens=len(coords),seconds=time.monotonic()-start,
                  shape_channels=32,latent_resolution=32,reference_view=0,texture_loaded=False))
            emit(index,seed,'shape_complete',tokens=len(coords));del result,features,context,noise
    del flow;release()


def decode(indices,seeds):
    pending=[]
    for i in indices:
        for s in seeds:
            path=folder(i,s)/'generation_complete.json'
            if not path.exists() or read(path).get('decoder_resolution')!=512:pending.append((i,s))
    if not pending:return
    decoder=load_model('shape_decoder')
    with torch.no_grad():
        for index,seed in pending:
            root=folder(index,seed);saved=torch.load(root/'original_latent.pt',weights_only=False)
            if (root/'generation_complete.json').exists():
                archive=root/'invalid_decoder256';archive.mkdir(exist_ok=True)
                for filename in ('generation_complete.json','native_original.obj'):
                    (root/filename).rename(archive/filename)
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            vertices,faces=decode_native(decoder,saved['features'].cuda(),saved['coords'].cuda())
            save_mesh(root/'native_original.obj',vertices,faces)
            write(root/'generation_complete.json',dict(time=now(),vertices=len(vertices),faces=len(faces),
                  decode_seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  native_single_view=True,gt_used=False,weights_frozen=True,decoder_resolution=decoder.resolution))
            emit(index,seed,'generation_complete',vertices=len(vertices),faces=len(faces));release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;snapshot('generate')
    indices=list(map(int,a.indices.split(',')));seeds=list(map(int,a.seeds.split(',')))
    encode(indices);sample_ss(indices,seeds);sample_shape(indices,seeds);decode(indices,seeds)


if __name__=='__main__':main()
