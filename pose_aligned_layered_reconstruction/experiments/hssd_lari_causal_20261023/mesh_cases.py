"""Decode registered existing SS cases with frozen SLat and native coordinates."""
import argparse
import json
import shutil
import numpy as np
import torch
from ...v2.dataset import SceneDataset, select_views
from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent
from ...mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import OUT, CELL, PILOT, INDICES, SEEDS, ARMS, read, verify, binding, write, freeze, locked_identity, check_record


@torch.no_grad()
def decode(arm, limit=None):
    import trimesh
    torch.set_num_threads(2)
    root=OUT/'mesh'/arm
    source_root=CELL/'evaluations'/f'occluded_{arm}'
    original=read(source_root/'identity.json')
    assert original==read(source_root/'ss_complete.json')
    if 'checkpoint' in original:verify(original['checkpoint'])
    identity={'protocol':freeze(),'arm':arm,'source_identity':binding(source_root/'identity.json'),
              'source_completion':binding(source_root/'ss_complete.json')}
    locked_identity(root/'identity.json',identity)
    dataset=SceneDataset(PILOT/'occluded16.json',require_labels=False)
    install_memory_efficient_mesh()
    runtime=None
    device=torch.device('cuda:0')
    added=0
    for index in INDICES:
        sample=select_views(dataset[index],torch.arange(8))
        folder=root/f'{index:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        gt_source=verify(sample['target_mesh_binding'])
        gt_path=folder/'gt.ply'
        if not gt_path.exists():shutil.copyfile(gt_source,gt_path)
        assert binding(gt_path)['sha256']==binding(gt_source)['sha256']
        target=trimesh.load(gt_path,force='mesh',process=False)
        for seed in SEEDS:
            dest=folder/f'{seed}.json'
            if dest.exists():
                check_record(read(dest),sample['uid'],seed)
                continue
            source_path=source_root/f'{index:03d}'/f'{seed}_ss.json'
            ss=read(source_path)
            check_record(ss,sample['uid'],seed)
            assert ss['arm']==arm and ss['index']==index and ss['gt']==sample['target_mesh_binding']
            row={'uid':sample['uid'],'object_uid':sample['object_uid'],'index':index,'seed':seed,'arm':arm,
                 'source_ss':binding(source_path),'gt':binding(gt_path),'success':False}
            if not ss['success']:
                row['failure']=ss['failure']
            else:
                with np.load(verify(ss['coords'])) as data:coords=data['coords']
                assert coords.ndim==2 and coords.shape[1]==4 and np.all(coords[:,0]==0)
                assert coords[:,1:].min()>=0 and coords[:,1:].max()<64
                if runtime is None:
                    runtime=load_frozen_slat30k(device)
                    locked_identity(root/'slat_binding.json',runtime['binding'])
                initial=sparse_noise(coords,master_noise(seed,index,device))
                z,_=sample_mesh_latent(runtime,sample,initial)
                mesh=runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
                if len(mesh.faces) and np.isfinite(mesh.vertices).all():
                    path=folder/f'{seed}.ply'
                    mesh.export(path)
                    mesh=trimesh.load(path,force='mesh',process=False)
                    row.update(success=True,mesh=binding(path),surface=surface_metrics(mesh,target),structure=structure_metrics(mesh),
                        native_bbox_min=np.asarray(mesh.vertices).min(0).tolist(),native_bbox_max=np.asarray(mesh.vertices).max(0).tolist(),
                        gt_bbox_min=np.asarray(target.vertices).min(0).tolist(),gt_bbox_max=np.asarray(target.vertices).max(0).tolist(),
                        fitted_transform=False)
                else:row['failure']='empty or nonfinite mesh'
                del initial,z,mesh
            write(dest,row)
            print(json.dumps({'arm':arm,'index':index,'seed':seed,'success':row['success'],'CD':row.get('surface',{}).get('chamfer_l1')}),flush=True)
            added+=1
            if limit is not None and added>=limit:return
    write(root/'complete.json',identity)


@torch.no_grad()
def galleries():
    import trimesh
    from PIL import Image, ImageDraw
    from ...data_v2.render_scenes import pack_scene, rasterize
    from ..hssd_pilot_20261020.scenes import look_at
    from ..hssd_pilot_20261020.visuals import shade
    device=torch.device('cuda:0')
    side=256
    focal=side*.5/np.tan(np.deg2rad(40)*.5)
    k=torch.tensor([[[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]]]*4,device=device,dtype=torch.float32)
    poses=[]
    for angle in np.deg2rad([45,135,225,315]):
        eye=2*np.array([np.cos(angle)*np.cos(np.pi/6),np.sin(angle)*np.cos(np.pi/6),.5])
        poses.append(look_at(eye,np.zeros(3),np.array([0.,0.,1.])))
    w=torch.tensor(np.array(poses),device=device,dtype=torch.float32)
    result=[]
    for index in INDICES:
        for seed in SEEDS:
            dest=OUT/'figures'/f'{index:03d}_{seed}.jpg'
            canvas=Image.new('RGB',(side*7,side*4+64),'white')
            draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f'Occluded | original index {index} | seed {seed} | native coordinates, identical fixed cameras',fill='black')
            paths=[OUT/'mesh'/'base'/f'{index:03d}'/'gt.ply']+[OUT/'mesh'/a/f'{index:03d}'/f'{seed}.ply' for a in ARMS]
            for col,(label,path) in enumerate(zip(['GT',*ARMS],paths)):
                draw.text((col*side+8,34),label,fill='black')
                if not path.exists():
                    draw.text((col*side+8,90),'FAILED / no mesh',fill='red')
                    continue
                mesh=trimesh.load(path,force='mesh',process=False)
                packed=pack_scene([shade(mesh)],[],device)
                render=rasterize(packed,k,w,image_size=(side,side),raster_side=side,layers=1,rgb=True,background_rgb=(1.,1.,1.))
                for view,pixels in enumerate(render['rgb']):
                    panel=Image.fromarray((pixels.numpy()*255).round().astype(np.uint8))
                    canvas.paste(panel,(col*side,view*side+64))
                del packed,render
            dest.parent.mkdir(exist_ok=True)
            canvas.save(dest,quality=94)
            result.append(binding(dest))
    return result


def summarize():
    protocol=freeze()
    dataset=SceneDataset(PILOT/'occluded16.json',require_labels=False)
    rows={}
    for arm in ARMS:
        root=OUT/'mesh'/arm
        assert read(root/'complete.json')==read(root/'identity.json')
        rows[arm]=[]
        for i in INDICES:
            for seed in SEEDS:
                r=read(root/f'{i:03d}'/f'{seed}.json');check_record(r,dataset.rows[i]['uid'],seed);rows[arm].append(r)
    result={'scope':'selected4 diagnostic cases; selection used prior SS outcomes, not a population performance estimate',
            'protocol':protocol,'records':rows,'comparisons':{}}
    pairs=[(a,'appearance_control') for a in ARMS if a not in ['base','appearance_control']]
    pairs += [('oracle_cell','oracle_point'),('predicted_cell','predicted_point'),('appearance_control','base')]
    for arm,control in pairs:
        records=[{**r,'mode':m} for a,m in [(arm,'candidate'),(control,'parent')] for r in rows[a]]
        result['comparisons'][arm+'_vs_'+control]=(aggregate_mesh_records(records,[dataset.rows[i]['uid'] for i in INDICES],SEEDS)
            if all(r['success'] for r in records) else {'failures':[r for r in records if not r['success']]})
    result['galleries']=galleries()
    write(OUT/'mesh_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['decode','summary']);p.add_argument('--arm',choices=ARMS);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.action=='summary':summarize()
    elif a.arm:decode(a.arm,a.limit)
    else:p.error('--arm required')
