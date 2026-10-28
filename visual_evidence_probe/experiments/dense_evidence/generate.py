"""Frozen30K baselines for preselected validation and real-pilot inputs."""
import argparse
import gc
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import OUT, ROUND6, PARAMS, registry, data_folder, folder, setup, read, write, emit, register
from .data import observations, model_input


def cleanup(): gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def encode(indices,device):
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder,_encode_patch_tokens
    pipeline=None
    for index in indices:
        root=data_folder(index); path=root/'features.pt'
        if path.exists(): continue
        rec, k, _, images, _=observations(index,device)
        if (root/'sample.pt').exists():
            sample=torch.load(root/'sample.pt',map_location='cpu',weights_only=False)
            features=sample['masked_visual_patch_features']
            if not torch.allclose(sample['intrinsics'].float(),k.cpu(),atol=1.e-4):
                raise ValueError('legacy real feature cameras do not match current input cameras')
        else:
            if pipeline is None: pipeline=build_frozen_dino_encoder(device=device)
            features=torch.cat([_encode_patch_tokens(pipeline,images[i:i+2]).half().cpu() for i in range(0,len(images),2)])
        torch.save(dict(features=features,intrinsics=k.cpu(),already_shared_preprocessed=True),path)
        print(dict(stage='features_ready',index=index,views=len(images)),flush=True)
    del pipeline; cleanup()


def fit_global(index,seed,mesh,k,w,masks,context,kind):
    """Use old input-only fitting bounds for clean inputs; fixed BOP frame for real."""
    from visual_evidence_probe.experiments.input_consistency.refine import transform,silhouettes
    from visual_evidence_probe.experiments.input_consistency.policy import select_checkpoint
    from visual_evidence_probe.render import mask_iou
    root=folder(index,seed)
    if kind=='real_scene':
        result=mesh.copy(); matrix=np.eye(4)
        write(root/'fit.json',dict(matrix=matrix.tolist(),uses_gt=False,
            policy='identity in provided annotated BOP object frame; no GT ICP or pose fitting',
            normalization_assistance='inherited BOP model bounds and object pose annotations'))
    else:
        v=torch.tensor(np.asarray(mesh.vertices,np.float32),device=k.device)
        faces=torch.tensor(np.asarray(mesh.faces,np.int32),device=k.device)
        target=F.interpolate(torch.tensor(masks,device=k.device,dtype=torch.float32)[:,None],size=(256,256),mode='area')[:,0]
        raw=torch.nn.Parameter(v.new_zeros(7)); optimizer=torch.optim.Adam([raw],lr=.03)
        snapshots=[]
        def current():
            rotation,translation,scale=transform(raw)
            return scale*(v@rotation.T)+translation
        def snap(step):
            with torch.no_grad():
                hard=silhouettes(current(),faces,k,w,context,518,soft=False).cpu().numpy()
                rotation,translation,scale=transform(raw)
                matrix=np.eye(4); matrix[:3,:3]=(scale*rotation).cpu().numpy(); matrix[:3,3]=translation.cpu().numpy()
                snapshots.append(dict(step=step,ious=mask_iou(hard,masks).tolist(),transform=matrix.tolist()))
        snap(0)
        for step in range(1,101):
            optimizer.zero_grad(set_to_none=True)
            pred=silhouettes(current(),faces,k,w,context,256)
            intersection=(pred*target).sum((1,2)); union=(pred+target-pred*target).sum((1,2)).clamp_min(1)
            loss=(1-intersection/union+.15*(pred*(1-target)).mean((1,2))+.08*(pred-target).abs().mean((1,2))).mean()
            loss.backward()
            if not torch.isfinite(raw.grad).all(): raise ValueError('nonfinite original-fit gradient')
            optimizer.step()
            if step%20==0: snap(step)
        chosen=select_checkpoint(snapshots,.005); matrix=np.array(chosen['transform'])
        result=mesh.copy(); result.apply_transform(matrix)
        write(root/'fit.json',dict(matrix=matrix.tolist(),uses_gt=False,chosen=chosen,snapshots=snapshots,
            policy='same bounded proper similarity fit to observed masks as earlier probes'))
    result.export(root/'original.obj',include_texture=False)


def copy_development(index,seeds):
    for seed in seeds:
        root=folder(index,seed); root.mkdir(parents=True,exist_ok=True)
        source=ROUND6/'objects'/f'{index:03d}'/str(seed)
        for name in ('original.obj','original_latent.pt'):
            path=root/name
            if not path.exists(): path.symlink_to(source/name)
        fit=read(source/'coordinate_frames.json')
        write(root/'fit.json',dict(matrix=fit['latent_native_to_mesh_world'],uses_gt=False,reused=True,source=str(source/'coordinate_frames.json')))
        write(root/'baseline_complete.json',dict(reused_round6=True,weights_frozen=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--indices',default=','.join(map(str,range(12,28))));p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe import runtime
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    from pose_aligned_layered_reconstruction.frozen_slat import sparse_noise,master_noise,sample_mesh_latent
    import nvdiffrast.torch as dr
    device=runtime.setup();register('generate');context=dr.RasterizeCudaContext(device=device)
    indices=list(map(int,a.indices.split(',')));seeds=list(map(int,a.seeds.split(',')))
    rows={r['index']:r for r in registry()['rows']}
    for index in indices:
        if rows[index]['split']=='development': copy_development(index,seeds)
    new=[i for i in indices if rows[i]['split']!='development']
    if not new:return
    encode(new,device)
    with FixedNative() as policy:
        ss=runtime.load_ss(device);write(OUT/f'ss_binding_{a.gpu}.json',ss['binding'])
        for index in new:
            sample=model_input(index)
            for seed in seeds:
                root=folder(index,seed);root.mkdir(parents=True,exist_ok=True)
                if not (root/'ss.npz').exists():runtime.sample_ss(ss,sample,index,seed,root/'ss.npz')
                emit(index,seed,'ss_ready')
        del ss;cleanup()
        slat=runtime.load_slat(device);write(OUT/f'slat_binding_{a.gpu}.json',slat['binding'])
        for index in new:
            sample=model_input(index)
            rec,k,w,_,masks=observations(index,device)
            for seed in seeds:
                root=folder(index,seed)
                if (root/'baseline_complete.json').exists():continue
                start=time.monotonic()
                with np.load(root/'ss.npz') as arrays:coords=arrays['coords']
                initial=sparse_noise(coords,master_noise(seed,index,device))
                z,stats=sample_mesh_latent(slat,sample,initial)
                mesh=slat['decoder'](z)[0].to_trimesh(transform_pose=False)
                mesh.export(root/'native_original.obj',include_texture=False)
                torch.save(dict(features=z.feats.cpu(),coords=z.coords.cpu()),root/'original_latent.pt')
                # Freeze latent before the independent, input-only global fit.
                fit_global(index,seed,mesh,k,w,masks,context,rec['kind'])
                write(root/'baseline_complete.json',dict(uid=rec['uid'],index=index,seed=seed,weights_frozen=True,
                    latent_saved=True,ss_fixed=True,sampler=stats,elapsed_seconds=time.monotonic()-start))
                emit(index,seed,'baseline_ready',seconds=time.monotonic()-start)
                del z,mesh;cleanup()
        write(OUT/f'generator_policy_{a.gpu}.json',policy.manifest())


if __name__=='__main__':main()
