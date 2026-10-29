import numpy as np
import torch
from pathlib import Path

from .config import BASE,OUT,PARAMS,case,original,read,write,digest,now
from ...geometry import observations,detached,load_mesh,render


def prepare():
    protocol=Path(__file__).with_name('PROTOCOL.md')
    identity=dict(params=PARAMS,base=str(BASE),arms=['latent_input','mesh_input','latent_oracle','mesh_oracle'],
                  protocol_sha256=digest(protocol),development_only=True)
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=identity:raise RuntimeError('diagnostic protocol changed')
    write(path,identity)
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            src=original(index,seed);root=case(index,seed);root.mkdir(parents=True,exist_ok=True)
            paths=[src/name for name in ('original_latent.pt','fit.json','evidence_sealed.npz','input_complete.json')]
            value=dict(index=index,seed=seed,inputs=[dict(path=str(p),sha256=digest(p)) for p in paths],
                       baseline_mesh=str(src/'aligned_base.obj'),baseline_mesh_bytes=(src/'aligned_base.obj').stat().st_size,
                       fixed_input_region=True,fixed_global_registration=True,gt_region_used=False)
            receipt=root/'binding.json'
            if receipt.exists() and read(receipt)!=value:raise RuntimeError('input binding changed')
            write(receipt,value)


def load(index,seed,context):
    from visual_evidence_probe.experiments.local_repair import photometric as photo
    from visual_evidence_probe.experiments.dense_evidence.matching import configured
    rec,k,w,images,masks=observations(index)
    root=original(index,seed);cache=case(index,seed)
    if (cache/'baseline_geometry.pt').exists():
        geometry=torch.load(cache/'baseline_geometry.pt',weights_only=False)
        vertices,faces=geometry['vertices'].cuda(),geometry['faces'].cuda()
    else:
        vertices,faces=load_mesh(root/'aligned_base.obj')
        torch.save(dict(vertices=vertices.cpu(),faces=faces.cpu()),cache/'baseline_geometry.pt')
    if (cache/'baseline_render.pt').exists():
        baseline={key:value.cuda() for key,value in torch.load(cache/'baseline_render.pt',weights_only=False).items()}
    else:
        baseline=detached(render(vertices,faces,k,w,context))
        torch.save({key:value.cpu() for key,value in baseline.items()},cache/'baseline_render.pt')
    with np.load(root/'evidence_sealed.npz') as a:
        region=torch.tensor(a['region'],device='cuda:0')
        active=torch.tensor(a['active'],device='cuda:0')
        train=a['train_views'].tolist();heldout=a['heldout_views'].tolist()
        target=detached(baseline)
        target['depth']=torch.tensor(a['target_depth'],device='cuda:0')
        target['depth_valid']=torch.tensor(a['target_valid'],device='cuda:0')
    _,gray,observed,eroded=photo.prepare_images(images,masks,PARAMS['side'])
    target.update(mask=target['depth_valid'].clone(),softmask=observed.float(),normal_valid=torch.zeros_like(observed))
    with configured(train,heldout[0]):
        supports=photo.validation_reference(baseline,gray,eroded,k,w)
        supports&=region[heldout[0]][None]>.5
    return dict(uid=rec['uid'],k=k,w=w,vertices=vertices,faces=faces,baseline=baseline,
                region=region,active=active,train=train,heldout=heldout,target=target,observed=observed,
                gray=gray,eroded=eroded,supports=supports)


def seal_inputs(index,seed):
    root=case(index,seed)
    paths=[root/arm/'complete.json' for arm in ('latent_input','mesh_input')]
    if not all(p.exists() for p in paths):raise RuntimeError('both input-only decisions must precede oracle loading')
    identity=dict(gt_used_for_input_decisions=False,arms=[dict(path=str(p),sha256=digest(p)) for p in paths])
    path=root/'input_sealed.json'
    if path.exists() and read(path)!=identity:raise RuntimeError('sealed input decisions changed')
    write(path,identity)


def oracle(index,seed,data,context):
    seal_inputs(index,seed)
    import trimesh
    desc=read(BASE/'data'/f'{index:03d}'/'evaluation_target.json')
    mesh=trimesh.load(desc['mesh'],force='mesh',process=False,skip_materials=True)
    mesh.apply_transform(np.asarray(desc['transform']))
    vertices=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0')
    faces=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0')
    with torch.no_grad():gt=detached(render(vertices,faces,data['k'],data['w'],context))
    # Keep the same INPUT-defined query mask. No GT filtering touches input arms.
    target=detached(gt);target['depth_valid']=data['target']['depth_valid']&gt['mask']
    target['normal_valid']=torch.zeros_like(gt['mask'])
    target['depth_valid'][data['heldout']]=False
    return target,vertices,faces


def evaluation_points(index):
    # Match the original evaluator exactly: sample the canonical scan in its
    # float64 trimesh representation, before the float32 rasterization cast.
    # Area-weighted face sampling is sensitive to early precision changes.
    import trimesh
    from ...metrics import surface_points
    desc=read(BASE/'data'/f'{index:03d}'/'evaluation_target.json')
    mesh=trimesh.load(desc['mesh'],force='mesh',process=False,skip_materials=True)
    mesh.apply_transform(np.asarray(desc['transform']))
    return surface_points(mesh.vertices,mesh.faces)
