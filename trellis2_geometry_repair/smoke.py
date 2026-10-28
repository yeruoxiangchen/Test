import argparse
import time

import numpy as np
import torch

from .config import OUT,PARAMS,folder,write,read,setup,snapshot,now
from .runtime import load_model,decode_native,FixedBranch,save_mesh,release
from .geometry import observations,render,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=5);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;snapshot('smoke')
    root=OUT/'validation/native_gradient'
    if (root/'result.json').exists() and not read(root/'result.json')['passed']:
        archive=OUT/'validation/debug'/f'native_recovery_failed_{now().replace(":","-")}'
        archive.parent.mkdir(parents=True,exist_ok=True);root.rename(archive)
    root.mkdir(parents=True,exist_ok=True)
    saved=torch.load(folder(a.index,a.seed)/'original_latent.pt',weights_only=False)
    features=saved['features'].cuda().float();coords=saved['coords'].cuda();std=saved['std'].cuda()
    decoder=load_model('shape_decoder',checkpoint=True);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    branch=FixedBranch(decoder,features,coords)
    probe=features.detach().clone().requires_grad_(True)
    v,f=branch(probe)
    max_error=float((v.detach()-branch.base_vertices).abs().max())
    if not torch.equal(f,branch.faces) or max_error>2.e-7:raise RuntimeError(f'native branch parity {max_error}')
    used=torch.unique(branch.faces.long())
    ids=used[::max(1,len(used)//4096)]
    weights=torch.sin(torch.arange(len(ids),device='cuda:0').float()+.3)
    scalar=(v[ids,2]*weights).mean()
    (scalar*PARAMS['loss_scale']).backward();gradient=probe.grad/PARAMS['loss_scale']
    if not bool(torch.isfinite(gradient).all()) or float(gradient.norm())<=0:raise RuntimeError('invalid native latent gradient')
    # Finite differences in native-std units on a fixed discrete branch.
    direction=torch.sign(gradient)*std;analytic=float((gradient*direction).sum())
    differences=[]
    with torch.no_grad():
        for eps in (.002,.01,.02):
            vp,_=branch(features+eps*direction);vm,_=branch(features-eps*direction)
            finite=float(((vp[ids,2]-vm[ids,2])*weights).mean()/(2*eps))
            relative=abs(finite-analytic)/max(abs(finite),abs(analytic),1.e-10)
            differences.append(dict(epsilon=eps,analytic=analytic,finite_difference=finite,relative_error=relative))
    gradient_ok=any(r['relative_error']<.25 and r['finite_difference']*analytic>0 for r in differences)
    write(root/'gradient.json',dict(time=now(),native_vertex_max_error=max_error,nonzero_gradient_norm=float(gradient.norm()),
          finite_differences=differences,passed=gradient_ok,discrete_topology_frozen=True))
    if not gradient_ok:raise RuntimeError(f'finite difference check failed: {differences}')
    del probe,v,scalar,gradient;release()
    # Reachable geometry target; actual native rendered meshes enter the loss.
    # The fixed-branch experiment remains a derivative diagnostic only.
    target=branch.base_vertices.detach();target_faces=branch.faces.detach();branch.close()
    active=coords[:,1].float()>=coords[:,1].float().median()
    generator=torch.Generator(device='cuda:0').manual_seed(PARAMS['random_seed'])
    corruption=torch.randn(features.shape,device='cuda:0',generator=generator)*.02*std*active[:,None]
    source=features+corruption;rows=torch.nonzero(active).flatten()
    delta=torch.nn.Parameter(torch.zeros((len(rows),32),device='cuda:0'))
    optimizer=torch.optim.Adam([delta],lr=.05)
    import nvdiffrast.torch as dr
    from visual_evidence_probe.experiments.editor_support.objective import geometry_loss
    context=dr.RasterizeCudaContext(device='cuda:0');_,k,w,_,_=observations(a.index)
    def mesh_loss(pred,target_render):
        return geometry_loss(pred,target_render,local_region=torch.ones_like(pred['depth']),
            depth_scale=.03,weights=dict(depth=1.,silhouette=.2,normal=.05,coverage=.2,protect=0.))['loss']
    with torch.no_grad():
        native_before_v,native_before_f=decode_native(decoder,source,coords)
        target_render=render(target,target_faces,k,w,context,128)
        original_render=render(native_before_v,native_before_f,k,w,context,128)
        initial=float(mesh_loss(original_render,target_render))
        domain=target_render['mask']|original_render['mask']
        native_error_before=float(error(original_render,target_render)[domain].mean())
        save_mesh(root/'target_generated.obj',target,target_faces)
        save_mesh(root/'corrupted_native.obj',native_before_v,native_before_f)
    def current():return source.index_add(0,rows,.05*std*torch.tanh(delta))
    curve=[];best_features=source.detach().clone();best_loss=initial
    for step in range(1,PARAMS['recovery_steps']+1):
        optimizer.zero_grad(set_to_none=True);v,f=decode_native(decoder,current(),coords)
        pred=render(v,f,k,w,context,128)
        raw_loss=mesh_loss(pred,target_render)
        if float(raw_loss.detach())<best_loss:
            best_loss=float(raw_loss.detach());best_features=current().detach().clone()
        loss=raw_loss/max(initial,1.e-8)
        (loss*PARAMS['loss_scale']).backward();delta.grad.div_(PARAMS['loss_scale'])
        if not bool(torch.isfinite(delta.grad).all()):raise RuntimeError('nonfinite recovery gradient')
        torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step();curve.append(float(loss.detach()))
        if step%5==0:print(dict(stage='reachable_recovery',step=step,relative_loss=curve[-1]),flush=True)
        del v,f,pred,loss,raw_loss
    with torch.no_grad():
        final=current().detach();native_after_v,native_after_f=decode_native(decoder,final,coords)
        last_loss=float(mesh_loss(render(native_after_v,native_after_f,k,w,context,128),target_render))
        if last_loss<best_loss:best_features=final;best_loss=last_loss
        final=best_features;native_after_v,native_after_f=decode_native(decoder,final,coords)
        save_mesh(root/'recovered_native.obj',native_after_v,native_after_f)
        after_render=render(native_after_v,native_after_f,k,w,context,128)
        native_error_after=float(error(after_render,target_render)[domain].mean())
    recovered=best_loss<initial and native_error_after<native_error_before*.95
    value=dict(time=now(),passed=bool(recovered),gradient_check=gradient_ok,native_parity_max_error=max_error,
               recovery=dict(native_geometry_loss_before=initial,native_geometry_loss_after=best_loss,curve=curve,
                             native_fixed_depth_before=native_error_before,native_fixed_depth_after=native_error_after),
               peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,
               synthetic_generated_target_only=True,scan_gt_used=False,teacher_latent_in_loss=False,
               same_ss=True,outside_rows_unchanged=bool(torch.equal(final[~active],source[~active])),
               optimization_path='native mesh rendered at every update',smoke_version=2)
    value['decoder_resolution']=decoder.resolution
    write(root/'result.json',value)
    if not recovered:raise RuntimeError(f'reachable geometry recovery failed: {value["recovery"]}')
    write(OUT/'validation/smoke_complete.json',value);print(value,flush=True)


if __name__=='__main__':main()
