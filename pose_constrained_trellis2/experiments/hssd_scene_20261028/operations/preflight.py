"""Independent CPU camera/crop/world checks on actual frozen pilot inputs."""
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
from pose_constrained_trellis2.experiments.hssd_scene_20261028.images import prepare_view

def project(vertices, world, camera, K):
    p=trimesh.transform_points(vertices,world)
    q=trimesh.transform_points(p,camera)
    uv=q@K.T
    return uv[:,:2]/uv[:,2:],q[:,2]

def main():
    results=[]
    for row in c.plan()['plan']['objects']:
        mesh=trimesh.load(row['mesh']['path'],process=False);v=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//2000)]
        center=(mesh.bounds[0]+mesh.bounds[1])/2;scale=.99999/np.ptp(mesh.bounds,axis=0).max()
        A=np.eye(4);A[:3,:3]*=scale;A[:3,3]=-center*scale
        world=np.linalg.inv(row['canonical_from_world']);native_world=world@np.linalg.inv(A)
        new=trimesh.transform_points(v,A)
        err=np.abs(trimesh.transform_points(v,world)-trimesh.transform_points(new,native_world)).max()
        if err>1e-8:raise AssertionError(('world_roundtrip',row['uid'],err))
        capture=c.read(row['capture_plan']['path'])['plan'];crop_errors=[];camera_errors=[]
        for view in row['views']:
            i=view['frame'];K=np.asarray(capture['intrinsics'][i]);E=np.asarray(capture['world_w2c'][i])
            np.testing.assert_allclose(E[:3,:3]@E[:3,:3].T,np.eye(3),atol=2e-6)
            if np.linalg.det(E[:3,:3])<.99999:raise AssertionError('Improper camera rotation')
            _,Knew,m=prepare_view(Path(view['rgb']['path']),Path(view['mask']['path']),torch.tensor(K,dtype=torch.float32),margin=.1,threshold=.8)
            uv,depth=project(v,world,E,K);native,_=project(new,native_world,E,K)
            valid=depth>.1
            if valid.any():camera_errors.append(float(np.abs(uv[valid]-native[valid]).max()))
            left,top,side,_=m['crop_xywh']
            # Pixel-centre resize/crop mapping, independent of production projection.
            expected=(uv-np.array([left,top])+.5)*(512/side)-.5
            actual,_=project(new,native_world,E,Knew.numpy())
            if valid.any():crop_errors.append(float(np.abs(expected[valid]-actual[valid]).max()))
        max_crop=max(crop_errors,default=0)
        # Absolute error bound avoids points at vanishing camera z dominating.
        if max_crop>0.02:raise AssertionError(('crop_projection',row['uid'],max_crop))
        results.append({'uid':row['uid'],'world_roundtrip':float(err),
            'camera_projection_max_px':max(camera_errors,default=0),'crop_projection_max_px':max_crop})
    print(json.dumps({'passed':True,'objects':len(results),'checks':results},indent=2))

if __name__=='__main__':main()
