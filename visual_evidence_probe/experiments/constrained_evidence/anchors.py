"""Actual decoded-mesh surface anchors; no corner-SDF proxy."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def closest_points(points,triangles):
    """Closest points on triangles; leading dimensions broadcast."""
    a,b,c=triangles.unbind(-2);ab=b-a;ac=c-a;ap=points-a
    normal=torch.cross(ab,ac,dim=-1);nn=normal.square().sum(-1).clamp_min(1.e-20)
    plane=points-(ap*normal).sum(-1)[...,None]/nn[...,None]*normal
    dot00=(ab*ab).sum(-1);dot01=(ab*ac).sum(-1);dot11=(ac*ac).sum(-1)
    dot20=((plane-a)*ab).sum(-1);dot21=((plane-a)*ac).sum(-1)
    denominator=(dot00*dot11-dot01.square()).clamp_min(1.e-20)
    u=(dot11*dot20-dot01*dot21)/denominator;v=(dot00*dot21-dot01*dot20)/denominator
    inside=(u>=0)&(v>=0)&(u+v<=1)&(normal.square().sum(-1)>1.e-20)
    candidates=[plane];distances=[torch.where(inside,(plane-points).square().sum(-1),torch.full_like(u,float('inf')))]
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;t=((points-start)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1.e-20)
        foot=start+t.clamp(0,1)[...,None]*edge
        candidates.append(foot);distances.append((foot-points).square().sum(-1))
    index=torch.stack(distances,-1).argmin(-1)
    return torch.stack(candidates,-2).gather(-2,index[...,None,None].expand(*index.shape,1,3)).squeeze(-2)


def prepare(mesh,selected,count,seed,device):
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tri=vertices[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)*.5
    eligible=np.flatnonzero(~np.asarray(selected,bool)&(area>1.e-14))
    if not len(eligible):raise ValueError('no exterior surface for three-dimensional anchors')
    rng=np.random.default_rng(seed);ids=rng.choice(eligible,count,replace=True,p=area[eligible]/area[eligible].sum())
    uv=rng.random((count,2));root=np.sqrt(uv[:,0]);bary=np.stack((1-root,root*(1-uv[:,1]),root*uv[:,1]),-1)
    points=(tri[ids]*bary[...,None]).sum(1)
    normals=cross[ids]/np.linalg.norm(cross[ids],axis=-1,keepdims=True).clip(1.e-20)
    return dict(points=torch.tensor(points,device=device,dtype=torch.float32),
        normals=torch.tensor(normals,device=device,dtype=torch.float32),
        source_faces=ids,barycentric=bary,source='area-weighted original mesh exterior; no GT')


def surface_terms(vertices,faces,anchors,neighbors=16,scale=1/256):
    """Differentiate the closest footpoint among K centroid-neighbor triangles.

    The discrete nearest-face search uses a detached current mesh. This is a
    K-neighbor surface-distance approximation, explicitly not a nearest-vertex
    surrogate or an exact all-triangle distance certificate.
    """
    triangles=vertices[faces.long()]
    with torch.no_grad():
        centroids=triangles.detach().mean(1).cpu().numpy()
        _,ids=cKDTree(centroids).query(anchors['points'].detach().cpu().numpy(),k=min(neighbors,len(centroids)),workers=2)
        if ids.ndim==1:ids=ids[:,None]
        ids=torch.tensor(ids,device=vertices.device,dtype=torch.long)
    choices=triangles[ids];query=anchors['points'][:,None]
    foot=closest_points(query,choices);distance=(foot-query).square().sum(-1)
    selected=distance.detach().argmin(-1);row=torch.arange(len(selected),device=vertices.device)
    closest=foot[row,selected];tri=choices[row,selected]
    normal=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
    normal=normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True).clamp_min(1.e-12)
    reference=anchors['normals'];normal_error=.5*torch.minimum((normal-reference).square().sum(-1),(normal+reference).square().sum(-1))
    displacement=torch.linalg.vector_norm(closest-anchors['points'],dim=-1)
    return dict(loss=((closest-anchors['points'])/scale).square().sum(-1).mean(),
        normal_loss=normal_error.mean(),mean=displacement.detach().mean(),
        p95=torch.quantile(displacement.detach(),.95))
