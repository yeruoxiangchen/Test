"""One codec phase/object per process; official T2 math with explicit world mapping."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from . import common as c

def row_for(uid):return next(r for r in c.plan()['plan']['objects'] if r['uid']==uid)
def folder(uid):
    p=c.DATA/'objects'/uid;p.mkdir(parents=True,exist_ok=True);return p

def dual(uid,resolution):
    import torch,trimesh,o_voxel
    row=row_for(uid);out=folder(uid)
    mesh=trimesh.load(row['mesh']['path'],force='mesh',process=False)
    v=torch.from_numpy(np.array(mesh.vertices)).float();f=torch.from_numpy(np.array(mesh.faces)).long()
    lo,hi=v.amin(0),v.amax(0);center=(lo+hi)/2;scale=0.99999/(hi-lo).max()
    A=np.eye(4);A[:3,:3]*=float(scale);A[:3,3]=(-center*scale).numpy()
    world=np.linalg.inv(np.asarray(row['canonical_from_world']))@np.linalg.inv(A)
    transformed=(v-center)*scale
    vnew=transformed.numpy();recovered=trimesh.transform_points(vnew,world)
    expected=trimesh.transform_points(mesh.vertices,np.linalg.inv(row['canonical_from_world']))
    error=float(np.abs(recovered-expected).max())
    if error>3e-5:raise ValueError('T2 normalization/world mismatch')
    c.immutable(out/'transform.json',{'old_to_t2':A.tolist(),'t2_to_world':world.tolist(),
       'roundtrip_max_abs':error,'source_mesh':row['mesh'],'normalization':'official 0.99999 extent, centered AABB'})
    trimesh.Trimesh(vnew,mesh.faces,process=False).export(out/'target_t2.ply')
    coords,vertices,intersected=o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=transformed,faces=f,grid_size=resolution,aabb=[[-.5,-.5,-.5],[.5,.5,.5]],
        face_weight=1.,boundary_weight=.2,regularization_weight=1e-2,timing=False)
    fractional=vertices*resolution-coords
    if not torch.all((fractional>=-1e-3)&(fractional<=1+1e-3)):raise ValueError('Invalid dual vertices')
    attr={'vertices':(fractional.clamp(0,1)*255).to(torch.uint8),
          'intersected':(intersected[:,:1]+2*intersected[:,1:2]+4*intersected[:,2:3]).to(torch.uint8)}
    o_voxel.io.write_vxz(str(out/f'dual_{resolution}.vxz'),coords,attr)
    c.write(out/f'dual_{resolution}.json',{'resolution':resolution,'voxels':len(coords)})

def encode(uid,resolution):
    import torch,o_voxel
    from trellis2 import models
    from trellis2.modules import sparse as sp
    out=folder(uid);coords,attr=o_voxel.io.read_vxz(str(out/f'dual_{resolution}.vxz'),num_threads=2)
    vertices=sp.SparseTensor((attr['vertices']/255.).float(),torch.cat([torch.zeros_like(coords[:,:1]),coords],-1))
    bits=attr['intersected'];inter=vertices.replace(torch.cat([bits%2,bits//2%2,bits//4%2],-1).bool())
    encoder=models.from_pretrained(c.models()['shape_encoder']).eval().cuda()
    with torch.no_grad(): z=encoder(vertices.cuda(),inter.cuda(),sample_posterior=False)
    if not torch.isfinite(z.feats).all():raise ValueError('Nonfinite Shape label')
    xyz=z.coords[:,1:].cpu().numpy().astype(np.int32);feats=z.feats.float().cpu().numpy()
    if (xyz<0).any() or (xyz>=resolution//16).any():raise ValueError('Invalid native support coordinates')
    if resolution==512 and len(xyz)>8192:raise ValueError(f'Native Shape512 has {len(xyz)} tokens; retain failure, do not truncate')
    np.savez_compressed(out/f'shape_{resolution}.npz',coords=xyz,feats=feats)
    c.write(out/f'encode_{resolution}.json',{'resolution':resolution,'tokens':len(xyz),'channels':feats.shape[1]})

def ss(uid):
    import torch
    from trellis2 import models
    out=folder(uid);xyz=np.load(out/'shape_1024.npz')['coords']
    x=torch.zeros(1,1,64,64,64,device='cuda');q=torch.from_numpy(xyz).long().cuda();x[0,0,q[:,0],q[:,1],q[:,2]]=1
    enc=models.from_pretrained(c.models()['ss_encoder']).eval().cuda()
    with torch.no_grad():z=enc(x,sample_posterior=False)
    del enc;gc.collect();torch.cuda.empty_cache()
    dec=models.from_pretrained(c.models()['ss_decoder']).eval().cuda()
    with torch.no_grad():y=dec(z)>0
    iou=float(((y&(x>0)).sum()/((y|(x>0)).sum().clamp_min(1))).item())
    if not torch.isfinite(z).all() or not y.any():raise ValueError('Invalid SS codec output')
    np.savez_compressed(out/'ss.npz',z=z[0].float().cpu().numpy())
    c.write(out/'ss.json',{'iou':iou,'target_cells':len(xyz),'decoded_cells':int(y.sum()),
        'source':'native_shape_1024_coordinates','finite':True})

def decode(uid):
    import torch,trimesh
    from trellis2 import models
    from trellis2.modules import sparse as sp
    from scipy.spatial import cKDTree
    out=folder(uid);pack=np.load(out/'shape_512.npz');xyz=torch.from_numpy(pack['coords']).int()
    z=sp.SparseTensor(torch.from_numpy(pack['feats']).cuda(),torch.cat([torch.zeros_like(xyz[:,:1]),xyz],-1).cuda())
    dec=models.from_pretrained(c.models()['shape_decoder']).eval().cuda();dec.set_resolution(512);dec.low_vram=True
    with torch.no_grad():m=dec(z)[0]
    v=m.vertices.float().cpu().numpy();faces=m.faces.cpu().numpy()
    if not len(v) or not len(faces) or not np.isfinite(v).all():raise ValueError('Invalid decoded mesh')
    mesh=trimesh.Trimesh(v,faces,process=False);mesh.export(out/'decoded_t2.ply')
    T=np.asarray(c.read(out/'transform.json')['t2_to_world']);world=mesh.copy();world.apply_transform(T);world.export(out/'decoded_world.ply')
    gt=trimesh.load(out/'target_t2.ply',process=False);np.random.seed(20261028)
    a=gt.sample(20000);b=mesh.sample(20000);da=cKDTree(b).query(a)[0];db=cKDTree(a).query(b)[0]
    metrics={'cd_l1':float((da.mean()+db.mean())/2),'cd_l2_squared':float((np.mean(da**2)+np.mean(db**2))/2)}
    for t in [.01,.02]:
        p=float((db<t).mean());r=float((da<t).mean());metrics[f'f@{t}']=2*p*r/max(p+r,1e-12)
    c.write(out/'codec.json',{'time':c.now(),'uid':uid,'vertices':len(v),'faces':len(faces),
        'metrics':metrics,'ss':c.read(out/'ss.json'),'world_transform':c.read(out/'transform.json'),
        'quality_review_required':True})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,7))
    for k,(az,el) in enumerate([(25,20),(130,20),(245,35)]):
        for j,pts in enumerate([a,b]):
            ax=fig.add_subplot(2,3,j*3+k+1,projection='3d');ax.scatter(*pts[::4].T,s=.3,c=pts[::4,2],cmap='viridis')
            ax.view_init(el,az);ax.set(xlim=(-.55,.55),ylim=(-.55,.55),zlim=(-.55,.55));ax.set_box_aspect((1,1,1));ax.set_title(['GT','T2 codec'][j]);ax.set_axis_off()
    fig.suptitle(uid+'  CD='+str(round(metrics['cd_l1'],5)));fig.tight_layout();fig.savefig(out/'codec.png',dpi=120);plt.close(fig)

def pose(uid):
    import torch
    from PIL import Image
    from .images import prepare_view
    from pose_constrained_trellis2.runtime import load_dinov3
    from pose_constrained_trellis2.constants import SAMPLE_FORMAT
    row=row_for(uid);out=folder(uid);capture=c.read(row['capture_plan']['path'])['plan']
    extractor=load_dinov3(device='cuda',model_name=c.models()['dino'])
    fs=[];ks=[];es=[];meta=[]
    for view in row['views']:
        i=view['frame'];K=torch.tensor(capture['intrinsics'][i],dtype=torch.float32)
        image,k,m=prepare_view(Path(view['rgb']['path']),Path(view['mask']['path']),K,margin=.1,threshold=.8)
        image.save(out/f'input_{i:03d}.png');features=extractor([image]).detach().half().cpu()
        fs.append(features);ks.append(k);es.append(capture['world_w2c'][i]);meta.append(m)
    features=torch.cat(fs)
    if tuple(features.shape[1:])!=(1029,1024) or not torch.isfinite(features).all():raise ValueError('DINO schema mismatch')
    payload={'format':SAMPLE_FORMAT,'uid':uid,'dino_features':features,'intrinsics':torch.stack(ks),
        'extrinsics':torch.tensor(es,dtype=torch.float32),'extrinsics_type':'w2c','camera_forward_sign':1.,
        'camera_image_y_sign':1.,'grid_transform':'identity','image_size':[512,512],
        'object_to_world':torch.tensor(c.read(out/'transform.json')['t2_to_world'],dtype=torch.float32),
        'frame_ids':[v['frame'] for v in row['views']],'scene_id':row['scene'],'instance_id':row['instance'],
        'source_images':[v['rgb']['path'] for v in row['views']], 'source_masks':[v['mask']['path'] for v in row['views']],
        'view_preprocessing':meta,'pose_source':'GT controlled protocol; not RGB-only estimated pose'}
    torch.save(payload,out/'pose.pt')
    # Verify camera projection against the actual mask in original full frames.
    from pose_constrained_trellis2.geometry import projection_geometry
    xyz=np.load(out/'shape_512.npz')['coords'];coords=torch.from_numpy(xyz).int()
    geo=projection_geometry(coords=coords,resolution=32,intrinsics=torch.tensor([capture['intrinsics'][v['frame']] for v in row['views']]),
        extrinsics=torch.tensor(es),image_height=2072,image_width=2072,object_to_world=payload['object_to_world'])
    c.write(out/'pose.json',{'views':len(fs),'features':list(features.shape),'source':'shared_scene_photographs',
        'projection_valid_fraction':float(geo['valid'].float().mean()),'mask_is_visible_not_amodal':True})

def finalize():
    from pose_constrained_trellis2.constants import CACHE_FORMAT
    from pose_constrained_trellis2.data import PoseTrellis2Dataset
    rows=[];metrics=[]
    for r in c.plan()['plan']['objects']:
        out=folder(r['uid']);metrics.append(c.read(out/'codec.json'))
        row={'uid':r['uid'],'split':r['split'],'scene':r['scene'],'view_count':len(r['views']),
             'pose_file':str(out/'pose.pt'),'ss_target_file':str(out/'ss.npz'),'shape_target_file':str(out/'shape_512.npz')}
        row.update({k+'_sha256':c.sha(row[k]) for k in ['pose_file','ss_target_file','shape_target_file']});rows.append(row)
    for subset in ['all','train','val','smoke']:
        path=c.DATA/f'{subset}.json';c.write(path,{'format':CACHE_FORMAT,'root':str(c.DATA),
            'plan_identity':c.plan()['identity'],'samples':rows[:1] if subset=='smoke' else [r for r in rows if subset=='all' or r['split']==subset]})
        for stage in ['ss','shape']:
            ds=PoseTrellis2Dataset(path,stage=stage)
            for i in range(len(ds)):ds[i]
    c.write(c.OUT/'codec_summary.json',{'time':c.now(),'objects':12,'training':8,'validation':4,
       'mean_cd_l1':float(np.mean([m['metrics']['cd_l1'] for m in metrics])),
       'minimum_ss_iou':min(m['ss']['iou'] for m in metrics),'records':metrics,
       'interpretation':'label roundtrip, not generative model quality; visual review required'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dual','encode','ss','decode','pose','finalize']);p.add_argument('--uid');p.add_argument('--resolution',type=int,default=512);a=p.parse_args()
    if a.action in ['dual','encode']:globals()[a.action](a.uid,a.resolution)
    elif a.action=='finalize':finalize()
    else:globals()[a.action](a.uid)
