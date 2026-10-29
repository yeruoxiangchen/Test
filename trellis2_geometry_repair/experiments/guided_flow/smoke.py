import time
import torch

from .config import OUT,PARAMS,case,previous,original,save,read,write,now,setup
from .backend import decode,protection
from .data import load,record,cpu
from .objective import routing_loss
from .guide import Guide
from .integration import integrate
from ...runtime import load_model,release
from ..local_regeneration.fields import to_mesh,lookup
from ..local_regeneration.sampling import join_original,restore_exterior
from ..local_regeneration.generation import sampler_configuration,noise_like

def main():
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    import nvdiffrast.torch as dr
    ctx=dr.RasterizeCudaContext(device='cuda:0');start=time.monotonic();index,seed=5,42
    saved=torch.load(original(index,seed)/'original_latent.pt',weights_only=False)
    coords=saved['coords'].cuda();features=saved['features'].cuda();mean=saved['mean'].cuda();std=saved['std'].cuda()
    decoder=load_model('shape_decoder',checkpoint=True,binding_root=OUT)
    data=load(index,seed,ctx);active=lookup(data['envelope'],coords[:,1:]).bool()
    with torch.no_grad():
        fields=decode(decoder,features,coords)
        expected=torch.load(previous(index,seed)/'base_fields.pt',weights_only=False)
        parity=float((fields['raw']-expected['raw'].cuda()).abs().max())
        if parity!=0 or not torch.equal(fields['coords'],expected['coords'].cuda()):raise RuntimeError('pre-threshold fields differ from original native decoder')
        v,f=to_mesh(fields)
        from ...runtime import decode_native
        nv,nf=decode_native(decoder,features,coords)
        if not torch.equal(v,nv) or not torch.equal(f,nf):raise RuntimeError('raw-field native mesh conversion differs')
        ref=protection(fields,data['envelope'],PARAMS['field_anchors'],PARAMS['subdivision_anchors'])
        sub=fields['subs'][0];allowed=lookup(data['envelope'],sub['coords']).bool()[:,None]&(sub['raw']<=0)
        ranked=torch.where(allowed,sub['raw'],torch.full_like(sub['raw'],-float('inf')))
        item=int(ranked.argmax());row=item//8;channel=item%8
        parent=sub['coords'][row];offset=parent.new_tensor([channel%2,(channel//2)%2,(channel//4)%2])
        target=((parent*2+offset).float()[None]+.5)/64-.5;before=float(sub['raw'][row,channel])
    del fields,expected,v,f,nv,nf;release()
    clean=(features-mean)/std;z=clean.detach().clone();created=False;curve=[]
    for step in range(3):
        with torch.enable_grad():
            leaf=z.detach().requires_grad_(True);fields=decode(decoder,leaf*std+mean,coords)
            value,_=routing_loss(fields['subs'][:1],target)
            gradient=torch.autograd.grad(value*512,leaf)[0]/512
        gradient[~active]=0
        if not bool(torch.isfinite(gradient).all()) or float(gradient.norm())==0:raise RuntimeError('missing-child route has no finite native latent gradient')
        scale=gradient[active].square().mean().sqrt().clamp_min(1.e-10)
        z=torch.where(active[:,None],z-.02*(gradient/scale).clamp(-3,3),clean)
        del leaf,fields,value,gradient;release()
        with torch.no_grad():
            fields=decode(decoder,z*std+mean,coords);after=float(fields['subs'][0]['raw'][row,channel])
        curve.append(dict(step=step+1,logit=after));del fields;release()
        if after>0:created=True;break
    if not created:raise RuntimeError('controlled absent native child did not become active')
    rec,metadata=record(data,ref,'input',index,seed)
    guide=Guide(decoder,torch.nn.Identity().cuda(),coords,mean,std,rec,ctx)
    result=guide(clean,active[:,None],11,.214285714)
    if not torch.equal(result[~active],clean[~active]):raise RuntimeError('real guide changed exterior')
    if not guide.curve[0]['gradient_norm']>0:raise RuntimeError('real geometry guidance was not exercised')
    save(OUT/'validation/real_guide.pt',dict(features=(result*std+mean).cpu(),coords=coords.cpu(),curve=guide.curve,GT_used=False))
    guide_curve=guide.curve
    del decoder,guide,rec,data,ref,features,result,z;release()
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    flow=load_model('shape_flow',binding_root=OUT);sampler=FlowEulerGuidanceIntervalSampler(PARAMS['sigma_min'])
    coords=torch.load(previous(index,seed)/(PARAMS['support']+'.pt'),weights_only=False)['coords'].cuda()
    original_features=saved['features'].cuda();old_coords=saved['coords'].cuda();idx,present=join_original(old_coords,coords)
    clean=torch.zeros((len(coords),32),device='cuda:0');clean[present]=(original_features[idx[present]]-mean)/std
    plan=torch.load(previous(index,seed)/'plan.pt',weights_only=False);active=lookup(torch.as_tensor(plan['roi']['envelope'],device='cuda:0'),coords[:,1:]).bool()
    condition=torch.load(__import__('trellis2_geometry_repair.config',fromlist=['OUT']).OUT/'data/005/reference_context.pt',weights_only=False)['features'].cuda()
    steps,rescale,kwargs=sampler_configuration('shape_slat');kwargs['neg_cond']=torch.zeros_like(condition);kwargs['_coords']=coords
    noise=noise_like(clean,index*100000+seed*1000+101)
    result=integrate(sampler,flow,clean,active[:,None],noise,condition,kwargs,steps=steps,rescale=rescale)
    features=restore_exterior(result*std+mean,original_features,old_coords,coords,active)
    expected=torch.load(previous(index,seed)/'candidates/ss_t065_n101_shape_n101.pt',weights_only=False)
    if not torch.equal(features.cpu(),expected['features']) or not torch.equal(coords.cpu(),expected['coords']):raise RuntimeError('zero-guide native Flow differs from V2')
    receipt=dict(time=now(),passed=True,native_field_max_difference=parity,native_mesh_exact=True,
        absent_child_before=before,absent_child_curve=curve,native_child_created=True,
        real_guide=read(OUT/'protocol.json')['params']['guide_steps'],real_geometry_update=metadata,real_guide_curve=guide_curve,
        real_guide_exterior_exact=True,zero_guide_v2_features_exact=True,zero_guide_v2_coords_exact=True,
        GT_used=False,controlled_route_target=True,seconds=time.monotonic()-start)
    receipt_path=OUT/'validation/smoke_complete.json'
    if receipt_path.exists():write(OUT/'validation/smoke_history'/f'{read(receipt_path)["time"].replace(":","-")}.json',read(receipt_path))
    write(receipt_path,receipt);print({k:v for k,v in receipt.items() if k!='real_guide_curve'},flush=True)

if __name__=='__main__':main()
