"""Post-hoc distance to frozen SS support; never feeds input decisions."""
import numpy as np
import torch

from .config import BASE,OUT,PARAMS,original,read,write,now,setup
from .data import seal_inputs
from ...geometry import render


def cell_union_distance(points,centers,half=1/64):
    """Exact Euclidean distance to a union of congruent axis-aligned cells."""
    points=np.asarray(points,np.float64);centers=np.asarray(centers,np.float64)
    values=[]
    for start in range(0,len(points),128):
        delta=np.maximum(np.abs(points[start:start+128,None]-centers[None])-half,0.)
        values.append(np.sqrt(np.square(delta).sum(-1).min(1)))
    return np.concatenate(values) if values else np.zeros(0)


def describe(points,centers,threshold):
    distance=cell_union_distance(points,centers)
    if not len(distance):return dict(points=0)
    return dict(points=len(distance),outside_fraction=float((distance>1.e-8).mean()),
                beyond_conservative_margin_fraction=float((distance>threshold).mean()),
                distance_mean=float(distance.mean()),distance_median=float(np.median(distance)),
                distance_p95=float(np.quantile(distance,.95)),distance_max=float(distance.max()))


def main():
    setup();torch.set_num_threads(2)
    # Every actual input decision is sealed before reading any extra GT here.
    for i in PARAMS['indices']:
        for seed in PARAMS['seeds']:seal_inputs(i,seed)
    import trimesh
    import nvdiffrast.torch as dr
    from visual_evidence_probe.experiments.editor_support.objective import world_from_depth
    context=dr.RasterizeCudaContext(device='cuda:0');records=[]
    margin=.5;threshold=np.sqrt(3)*(2+2*margin)/512
    with torch.no_grad():
        for index in PARAMS['indices']:
            obs=read(BASE/'data'/f'{index:03d}'/'observations.json')
            desc=read(BASE/'data'/f'{index:03d}'/'evaluation_target.json')
            mesh=trimesh.load(desc['mesh'],force='mesh',process=False,skip_materials=True)
            mesh.apply_transform(np.asarray(desc['transform']))
            k=torch.tensor(obs['intrinsics'],device='cuda:0');w=torch.tensor(obs['extrinsics'],device='cuda:0')
            v=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0');f=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0')
            gt=render(v,f,k,w,context)
            for seed in PARAMS['seeds']:
                src=original(index,seed);saved=torch.load(src/'original_latent.pt',weights_only=False)
                centers=(saved['coords'][:,1:].numpy()+.5)/32-.5
                inverse=np.linalg.inv(np.asarray(read(src/'fit.json')['matrix']))
                with np.load(src/'evidence_sealed.npz') as a:
                    query=torch.tensor(a['target_valid'],device='cuda:0')&gt['mask']
                    teacher=torch.tensor(a['target_depth'],device='cuda:0')
                teacher_world=world_from_depth(teacher,k,w)[query].cpu().numpy()
                gt_world=gt['xyz'][query].cpu().numpy()
                convert=lambda x:x@inverse[:3,:3].T+inverse[:3,3]
                record=dict(index=index,seed=seed,uid=obs['uid'],support_cells=len(centers),
                       teacher=describe(convert(teacher_world),centers,threshold),
                       gt=describe(convert(gt_world),centers,threshold),
                       frozen_fit=True,gt_used_for_input_training=False)
                records.append(record);print(record,flush=True)
    write(OUT/'support_audit.json',dict(time=now(),posthoc_diagnostic=True,records=records,
          support_grid=32,decoder_grid=512,voxel_margin=margin,conservative_distance_threshold=threshold,
          explanation='Distance to original occupied coarse-cell union after the SAME frozen inverse registration. '
          'Subdivisions remain in original parent cells; the generous margin also covers local adjacent-cell triangles. '
          'Outside support can reflect global misregistration or coarse geometry, which this audit does not separate. '
          'Inside support does not imply an optimizer can reach the target. No GT correction is fed back.'))


if __name__=='__main__':main()
