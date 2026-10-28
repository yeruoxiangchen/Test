import argparse
import gc

import numpy as np
import torch

from .config import OUT,PARAMS,folder,read,write,setup,snapshot,emit,now
from .geometry import observations,load_mesh,render,error,novel_cameras
from .metrics import surface_points,surface_scores


def one(index,context):
    import trimesh
    if not all((folder(index,s)/'input_complete.json').exists() for s in PARAMS['seeds']):
        raise RuntimeError('seal both seeds before GT evaluation')
    rec,k,w,_,_=observations(index)
    desc=read(OUT/'data'/f'{index:03d}'/'evaluation_target.json')
    mesh=trimesh.load(desc['mesh'],force='mesh',process=False,skip_materials=True)
    mesh.apply_transform(np.asarray(desc['transform']))
    target_points=surface_points(mesh.vertices,mesh.faces)
    nk,nw=novel_cameras(k.cpu().numpy(),w.cpu().numpy())
    write(OUT/'data'/f'{index:03d}'/'evaluation_cameras.json',dict(intrinsics=nk.tolist(),extrinsics=nw.tolist()))
    nk=torch.tensor(nk,device='cuda:0');nw=torch.tensor(nw,device='cuda:0')
    gv=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0');gf=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0')
    gt=render(gv,gf,nk,nw,context)
    torch.save({key:gt[key].cpu() for key in ('depth','mask','xyz','normal')},OUT/'data'/f'{index:03d}'/'gt_evaluation_render.pt')
    for seed in PARAMS['seeds']:
        root=folder(index,seed)
        if (root/'result.json').exists():continue
        selection=read(root/'input_complete.json')
        bv,bf=load_mesh(root/'aligned_base.obj');base=render(bv,bf,nk,nw,context)
        with np.load(root/'evidence_sealed.npz') as a:selected=torch.tensor(a['selected_faces'],device='cuda:0')
        region=selected[base['face_id'].clamp_min(0)]&base['mask']
        domain=base['mask']|gt['mask'];base_error=error(base,gt)
        bad=domain&(base_error>=1/256);good=domain&~bad
        masks=dict(all=domain,bad=bad,good=good,selected_bad=region&bad)
        def score(vertices,faces):
            pred=render(vertices,faces,nk,nw,context);errors=error(pred,gt)
            values={name:float(errors[mask].mean()) if bool(mask.any()) else None for name,mask in masks.items()}
            return dict(surface=surface_scores(vertices.cpu().numpy(),faces.cpu().numpy(),target_points),
                        novel_depth=values,bad_pixels=int(bad.sum()),good_pixels=int(good.sum()))
        baseline=score(bv,bf);scores={};cache={(root/'aligned_base.obj').resolve():baseline}
        # step0 is the same native decode plus the exact same fit.
        cache[(root/'step_000.obj').resolve()]=baseline
        for name,path in [('raw',root/'native_original.obj')]+[(name,root/f'{name}.obj') for name in ('accepted','candidate','proposal')]:
            path=path.resolve()
            if path not in cache:
                v,f=load_mesh(path);cache[path]=score(v,f)
            scores[name]=cache[path]
        value=dict(time=now(),index=index,seed=seed,uid=rec['uid'],original=baseline,outputs=scores,
                   accepted_edit=selection['accepted_edit'],fit=read(root/'fit.json'),
                   timing_seconds=selection['seconds'],peak_gib=selection['peak_gib'],
                   decisions=selection['decisions'],gt_evaluation_only=True,no_additional_alignment=True)
        write(root/'result.json',value);emit(index,seed,'evaluated_complete');gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');a=p.parse_args()
    setup();torch.set_num_threads(2);snapshot('evaluate')
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0')
    with torch.no_grad():
        for index in map(int,a.indices.split(',')):one(index,context)


if __name__=='__main__':main()
