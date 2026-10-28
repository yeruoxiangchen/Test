"""Frozen RGB baselines on common views; native/world meshes, no LaRI branch."""
import argparse
import gc
import numpy as np
import torch
from .common import *
from ...v2.dataset import SceneDataset, select_views
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import build_ss
from ...native_ss import LayeredSSCFGFlow
from ...checkpoints import DEFAULT_PARENT_SS30K
from ...pretrained import DEFAULT_PRETRAINED, load_trellis_model
from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent
from ...evaluate_ss_occupancy import occupancy_metrics
from ...mesh_metrics import surface_metrics, structure_metrics, sample_surface
from ..hssd_cell_query_20261023.model import build, ResidualCFG, observations
from ..hssd_operator_audit_20261025.kernel_policy import FixedNative
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh


def folder(arm, uid, seed):
    return OUT / 'evaluations' / arm / uid / str(seed)


def run_identity():
    frozen = plan()
    result = {'plan_identity': frozen['identity'], 'manifest': binding(OUT/'manifest.json'),
        'checkpoint': binding(CELL/'train/occluded_appearance_control/last.pt'),
        'sources': [binding(p) for p in sorted(HERE.glob('*.py'))] +
                   [binding(HERE/'run.sh'), binding(HERE.parent/'hssd_operator_audit_20261025/kernel_policy.py'),
                    binding(HERE.parent/'round2_v2/render_mesh_audit.py')],
        'ss_sampling': {'steps':25,'cfg_strength':5.,'cfg_interval':[.5,1.],'rescale_t':3.},
        'input_frames':FRAMES,'seeds':SEEDS,'arms':ARMS,
        'scope':'baseline transfer to new shared views of old training objects; unequal training history is not a causal method ablation'}
    locked(OUT/'evaluation_identity.json', result)
    return identity(result)


@torch.no_grad()
def sample_ss(arm, limit=None):
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    ident = run_identity()
    dataset = SceneDataset(OUT/'manifest.json', require_labels=False, require_ss_target=True)
    cases = [(i,s) for i in range(len(dataset)) for s in SEEDS]
    if limit is not None:
        cases = cases[:limit]
    todo=[]
    for i,s in cases:
        path=folder(arm,dataset.rows[i]['object_uid'],s)/'ss.json'
        if path.exists():
            record=read(path)
            if record['identity'] != ident: raise ValueError('Changed SS evaluation')
            verify(record['coords'])
        else: todo.append((i,s))
    if not todo: return
    event('load_ss', arm=arm)
    if arm == 'yxc30k':
        sampler, model, bindings = build_ss(GeometryConfig(),device,str(DEFAULT_PARENT_SS30K),
            str(DEFAULT_PRETRAINED),False,condition_mode='no_layer')
        pretrained = str(DEFAULT_PRETRAINED)
    else:
        checkpoint=torch.load(CELL/'train/occluded_appearance_control/last.pt',map_location='cpu',weights_only=False)
        contract=checkpoint['contract']
        if contract['arm'] != 'appearance_control' or checkpoint['step'] != contract['steps']:
            raise ValueError('Expected completed appearance-control checkpoint')
        sampler,model,saved,bindings=build(contract,device)
        if checkpoint['bindings'] != bindings: raise ValueError('Appearance base binding mismatch')
        model.residual.load_state_dict(checkpoint['residual'],strict=True)
        pretrained=saved['pretrained']
    model.eval().requires_grad_(False)
    decoder, decoder_binding=load_trellis_model(pretrained,'ss_dec_conv3d_16l8_fp16')
    decoder=decoder.to(device).eval().requires_grad_(False)
    for i,seed in todo:
        source=dataset[i]
        row=plan()['plan']['objects'][i]
        sample=select_views(observations(source),torch.arange(len(FRAMES)))
        positive=sample['stock_condition'].to(device)
        dest=folder(arm,source['object_uid'],seed)
        dest.mkdir(parents=True,exist_ok=True)
        event('sample_ss',arm=arm,uid=source['object_uid'],seed=seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if arm=='yxc30k':
                wrapper=LayeredSSCFGFlow(model,positive,sample,use_layer_branch=False)
            else:
                prepared=model.prepare(sample,device)
                wrapper=ResidualCFG(model,positive,prepared,enabled=True)
            gen=torch.Generator(device=device).manual_seed(seed*1000003+row['original_index']*1009)
            noise=torch.randn((1,8,16,16,16),generator=gen,device=device)
            latent=sampler.sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
            logits=decoder(latent.to(next(decoder.parameters()).dtype))
        if not torch.isfinite(logits).all() or not torch.isfinite(latent).all():
            raise ValueError('Nonfinite SS; target retained as failure')
        coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
        if not len(coords): raise ValueError('Empty SS; target retained as failure')
        np.savez_compressed(dest/'ss.npz',coords=coords)
        write(dest/'ss.json',{'identity':ident,'uid':source['uid'],'object_uid':source['object_uid'],
            'seed':seed,'arm':arm,'coords':binding(dest/'ss.npz'),'gt':row['mesh'],
            'input_sample':binding(OUT/'objects'/source['object_uid']/'sample.pt'),
            'model_binding':bindings,'decoder_binding':decoder_binding,
            'occupancy':occupancy_metrics(coords,source['target_coords'].numpy()),'success':True})
    event('ss_complete',arm=arm,cases=len(cases))


def partitioned_metrics(mesh,target,row,p):
    """Evaluation-only visible/occluded GT partitions; approximate raster tolerance."""
    from scipy.spatial import cKDTree
    points,_=sample_surface(target,20000,42)
    predicted,_=sample_surface(mesh,20000,42)
    a=np.asarray(row['canonical_from_world'])
    inv=np.linalg.inv(a)
    world=(np.c_[points,np.ones(len(points))]@inv.T)[:,:3]
    pred_world=(np.c_[predicted,np.ones(len(predicted))]@inv.T)[:,:3]
    visible=np.zeros(len(points),bool)
    occluded=np.zeros(len(points),bool)
    free=np.zeros(len(predicted),bool)
    tolerance=.02/np.linalg.norm(a[:3,0])
    for i in FRAMES:
        depth=np.load(OUT/'evaluation_gt'/f'{i:03d}.npz')
        k,w=np.array(p['intrinsics'][i]),np.array(p['world_w2c'][i])
        for cloud,kind in [(world,'gt'),(pred_world,'pred')]:
            camera=np.c_[cloud,np.ones(len(cloud))]@w.T
            z=camera[:,2]
            uv=camera[:,:3]@k.T
            pixels=np.rint(uv[:,:2]/np.maximum(z[:,None],1.e-8)).astype(np.int64)
            valid=(z>0)&(pixels[:,0]>=0)&(pixels[:,0]<518)&(pixels[:,1]>=0)&(pixels[:,1]<518)
            xx,yy=pixels[:,0].clip(0,517),pixels[:,1].clip(0,517)
            observed=depth['depth_z_world'][yy,xx]
            owner=depth['instance_id'][yy,xx]
            if kind=='gt':
                visible |= valid&(owner==row['instance']+2)&(np.abs(z-observed)<=tolerance)
                occluded |= valid&(observed>0)&(owner!=row['instance']+2)&(observed<z-tolerance)
            else:
                free |= valid&(observed>0)&(z<observed-tolerance)
    distances=cKDTree(predicted).query(points,workers=2)[0]
    groups={'visible':visible,'externally_occluded':~visible&occluded,'other_unobserved':~visible&~occluded}
    result={}
    for name,mask in groups.items():
        result[name]={'gt_samples':int(mask.sum()),'gt_to_pred_mean':float(distances[mask].mean()) if mask.any() else None,
                      'recall_0p02':float((distances[mask]<.02).mean()) if mask.any() else None}
    result['predicted_surface_free_space_violation_fraction']=float(free.mean())
    result['visibility_depth_tolerance_canonical']=.02
    result['scope']='GT-assisted diagnostic; empty strata are null; not a collision-volume metric'
    return result


@torch.no_grad()
def mesh(limit=None):
    import trimesh
    torch.set_num_threads(2)
    ident=run_identity()
    p=plan()['plan']
    dataset=SceneDataset(OUT/'manifest.json',require_labels=False)
    cases=[(arm,i,s) for arm in ARMS for i in range(len(dataset)) for s in SEEDS]
    if limit is not None: cases=cases[:limit]
    runtime=None
    install_memory_efficient_mesh()
    device=torch.device('cuda:0')
    with FixedNative() as fixed:
        for arm,i,seed in cases:
            row=p['objects'][i]
            dest=folder(arm,row['uid'],seed)
            result=dest/'mesh.json'
            if result.exists():
                record=read(result)
                if record['identity']!=ident: raise ValueError('Changed completed mesh identity')
                verify(record['mesh']);verify(record['world_mesh'])
                continue
            ss=read(dest/'ss.json')
            if ss['identity']!=ident: raise ValueError('SS identity mismatch')
            if runtime is None:
                event('load_slat')
                runtime=load_frozen_slat30k(device,evaluation_uids=[r['uid'] for r in p['objects']])
            event('sample_mesh',arm=arm,uid=row['uid'],seed=seed)
            source=dataset[i]
            sample=select_views(observations(source),torch.arange(len(FRAMES)))
            with np.load(verify(ss['coords'])) as data: coords=data['coords']
            initial=sparse_noise(coords,master_noise(seed,row['original_index'],device))
            latent,calls=sample_mesh_latent(runtime,sample,initial)
            if (calls['positive_calls'],calls['negative_calls'])!=(25,19):
                raise ValueError('Incomplete SLat sampling')
            output=runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
            if not len(output.faces) or not np.isfinite(output.vertices).all():
                raise ValueError('Empty/nonfinite mesh; no target dropped')
            output.export(dest/'mesh_canonical.ply')
            output=trimesh.load(dest/'mesh_canonical.ply',force='mesh',process=False)
            target=trimesh.load(verify(row['mesh']),force='mesh',process=False)
            world=output.copy();world.apply_transform(np.linalg.inv(row['canonical_from_world']))
            world.export(dest/'mesh_world.ply')
            write(result,{'identity':ident,'uid':source['uid'],'object_uid':row['uid'],'arm':arm,'seed':seed,
                'mesh':binding(dest/'mesh_canonical.ply'),'world_mesh':binding(dest/'mesh_world.ply'),
                'gt':row['mesh'],'ss':binding(dest/'ss.json'),'success':True,'fitted_transform':False,
                'canonical_from_world':row['canonical_from_world'],'slat_binding':runtime['binding'],
                'calls':calls,'kernel_policy':fixed.manifest(),'surface':surface_metrics(output,target),
                'structure':structure_metrics(output),'partitioned':partitioned_metrics(output,target,row,p)})
            del initial,latent,output,target,world
    event('mesh_complete',cases=len(cases))


def summary():
    import trimesh
    from .visuals import make_figures
    p=plan()['plan']
    ident=run_identity()
    rows=[read(folder(a,r['uid'],s)/'mesh.json') for a in ARMS for r in p['objects'] for s in SEEDS]
    if len(rows)!=24 or not all(r['success'] for r in rows): raise ValueError('Need all 24 results')
    for row in rows:
        if row['identity']!=ident or row['fitted_transform']:raise ValueError('Mixed evaluation protocol')
        for key in ['mesh','world_mesh','ss','gt']:verify(row[key])
    means={a:{k:float(np.mean([r['surface'][k] for r in rows if r['arm']==a]))
        for k in ['chamfer_l1','fscore_0p02']} for a in ARMS}
    world_dir=OUT/'world';world_dir.mkdir(exist_ok=True)
    for arm in ARMS:
        for seed in SEEDS:
            scene=trimesh.Scene()
            for row in rows:
                if (row['arm'],row['seed'])==(arm,seed):
                    scene.add_geometry(trimesh.load(verify(row['world_mesh']),process=False),node_name=row['object_uid'])
            scene.export(world_dir/f'{arm}_{seed}.glb')
    figures=make_figures(rows)
    result={'time':now(),'means':means,'records':rows,'scope':p['scope'],'conditioning':p['conditioning'],
        'figures':figures,
        'world_export':'four independently generated target meshes; fixed GT-derived placements, not estimated layout',
        'external_gena_included':False,'training_launched':False}
    write(OUT/'summary.json',result)
    lines=['# SharedScene4 baseline results','','All 24 meshes completed. No new training.',
        '',p['scope'],'',p['conditioning'],'','| Arm | CD ↓ | F@0.02 ↑ |','|---|---:|---:|']
    lines += [f"| {a} | {means[a]['chamfer_l1']:.9f} | {means[a]['fscore_0p02']:.6f} |" for a in ARMS]
    lines += ['', 'GENA3D is not included in this table. These checkpoints have different training histories; this is a baseline transfer diagnostic, not evidence for a new contribution.',
        '', '[Input sheet](shared_input_contact_sheet.jpg); [native mesh comparisons](figures/); [all records and visibility partitions](summary.json); [world meshes](world/).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event('baseline_complete_review_required',meshes=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['ss','mesh','summary'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.phase=='ss':
        if args.arm is None: parser.error('--arm is required')
        sample_ss(args.arm,args.limit)
    elif args.phase=='mesh':mesh(args.limit)
    else:summary()
