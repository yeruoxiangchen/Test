import numpy as np
from scipy.spatial import cKDTree


def surface_points(vertices,faces,count=50000,seed=20261028):
    tri=np.asarray(vertices)[np.asarray(faces,np.int64)]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    rng=np.random.default_rng(seed);ids=rng.choice(len(tri),size=count,p=area/area.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    points=(1-u[:,None])*tri[ids,0]+(u*(1-v))[:,None]*tri[ids,1]+(u*v)[:,None]*tri[ids,2]
    return points


def surface_scores(vertices,faces,target_points):
    points=surface_points(vertices,faces)
    d1=cKDTree(target_points).query(points,workers=2)[0]
    d2=cKDTree(points).query(target_points,workers=2)[0]
    value=dict(chamfer_l1=float(.5*(d1.mean()+d2.mean())))
    for t in (.01,.02,.05):
        p,r=float((d1<t).mean()),float((d2<t).mean())
        value[f'fscore_{str(t).replace(".","p")}']=2*p*r/max(p+r,1.e-12)
    return value
