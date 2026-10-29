"""Native GPU technical gates; no GT or reconstruction-quality claim."""
import time
import torch
import torch.nn.functional as F

from .config import BASE,OUT,PARAMS,case,source,read,write,save_tensor,setup,now
from .sampling import masked_euler,restore_exterior
from .fields import decode_fields,to_mesh,compose
from .generation import sampler_configuration,noise_like,ss_coordinates
from ...runtime import load_model,sparse,release


def main():
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.7);start=time.monotonic()
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    index,seed=5,42;root=case(index,seed);root.mkdir(parents=True,exist_ok=True)
    saved=torch.load(source(index,seed)/'original_latent.pt',weights_only=False)
    features=saved['features'].cuda();coords=saved['coords'].cuda();mean=saved['mean'].cuda();std=saved['std'].cuda()
    decoder=load_model('shape_decoder',binding_root=OUT)
    with torch.no_grad():
        v,f,fields=decode_fields(decoder,features,coords);rv,rf=to_mesh(fields)
        parity=float((v-rv).abs().max());faces_equal=torch.equal(f,rf)
        if parity!=0 or not faces_equal:raise RuntimeError('captured native FDG conversion parity failed')
        identity,checks=compose(fields,fields,torch.zeros((32,32,32),device='cuda:0'))
        iv,iff=to_mesh(identity)
        if not torch.equal(v,iv) or not torch.equal(f,iff):raise RuntimeError('empty edit changed native geometry')
        alpha=torch.zeros((32,32,32),device='cuda:0');c=coords[len(coords)//2,1:].long();alpha[c[0],c[1],c[2]]=1
        altered=dict(coords=fields['coords'],raw=fields['raw']+.01)
        _,exterior_checks=compose(fields,altered,alpha)
    del decoder,v,f,rv,rf,iv,iff,fields,identity,altered;release()
    flow=load_model('shape_flow',binding_root=OUT);sampler=FlowEulerGuidanceIntervalSampler(PARAMS['sigma_min'])
    context=torch.load(BASE/'data'/f'{index:03d}'/'reference_context.pt',weights_only=False)['features'].cuda()
    steps,rescale,kwargs=sampler_configuration('shape_slat');kwargs['neg_cond']=torch.zeros_like(context)
    clean=(features-mean)/std;active=torch.zeros(len(coords),device='cuda:0',dtype=torch.bool);active[len(coords)//2]=True
    with torch.no_grad():
        result=masked_euler(sampler,flow,clean,active[:,None],noise_like(clean,9871),.35,steps,rescale,context,
                     wrap=lambda x:sparse(x,coords),unwrap=lambda x:x.feats,kwargs=kwargs)
        restored=restore_exterior(result*std+mean,features,coords,coords,active)
        if not torch.equal(restored[~active],features[~active]):raise RuntimeError('real Shape exterior drift')
        changed=float((restored[active]-features[active]).abs().max())
        if not changed>0:raise RuntimeError('local Shape smoke did not exercise generation')
    save_tensor(OUT/'validation/local_shape.pt',dict(features=restored.cpu(),coords=coords.cpu(),gt_used=False))
    del flow,features,coords,clean,result,restored;release()
    flow=load_model('ss_flow',binding_root=OUT);decoder=load_model('ss_decoder',binding_root=OUT)
    steps,rescale,kwargs=sampler_configuration('sparse_structure');kwargs['neg_cond']=torch.zeros_like(context)
    with torch.no_grad():
        torch.manual_seed(seed);noise=torch.randn((1,8,16,16,16),device='cuda:0')
        clean=sampler.sample(flow,noise,cond=context,steps=steps,rescale_t=rescale,verbose=False,**kwargs).samples
        replayed=ss_coordinates(decoder,clean)
        expected=torch.load(source(index,seed)/'ss.pt',weights_only=False)['coords'].cuda()
        if not torch.equal(replayed,expected):raise RuntimeError('SS original coordinate replay failed')
        save_tensor(root/'ss_dense_replay.pt',dict(latent=clean.cpu(),coords=replayed.cpu(),native_replay_exact=True))
        mask=torch.zeros((1,1,16,16,16),device='cuda:0',dtype=torch.bool)
        parent=(expected[len(expected)//2,1:]//2).long();mask[0,0,parent[0],parent[1],parent[2]]=True
        local=masked_euler(sampler,flow,clean,mask,noise_like(clean,9872),.65,steps,rescale,context,kwargs=kwargs)
        if not torch.equal(local[~mask.expand_as(local)],clean[~mask.expand_as(clean)]):raise RuntimeError('SS exterior replay failed')
    save_tensor(OUT/'validation/local_ss.pt',dict(latent=local.cpu(),gt_used=False,synthetic_region=True))
    write(OUT/'validation/smoke_complete.json',dict(time=now(),passed=True,native_vertex_parity_max=parity,
        native_faces_exact=faces_equal,empty_region_mesh_exact=True,exterior_fields_exact=exterior_checks['exterior_fields_exact'],
        real_shape_exterior_exact=True,real_shape_changed=changed,ss_original_coords_exact=True,
        real_ss_exterior_exact=True,gt_used=False,synthetic_region_only=True,seconds=time.monotonic()-start))
    print(read(OUT/'validation/smoke_complete.json'),flush=True)


if __name__=='__main__':main()
