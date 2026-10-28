"""Deterministic native FDG extraction using exact integer coordinate lookup."""
import torch

OFFSETS=(((0,0,0),(0,0,1),(0,1,1),(0,1,0)),
         ((0,0,0),(1,0,0),(1,0,1),(0,0,1)),
         ((0,0,0),(0,1,0),(1,1,0),(1,0,0)))


def extract(coords,dual_vertices,intersected_flag,split_weight,aabb,grid_size,method='sorted'):
    grid=torch.as_tensor(grid_size,device=coords.device,dtype=torch.int64)
    if grid.ndim==0:grid=grid.repeat(3)
    if bool(((coords<0)|(coords>=grid)).any()):raise RuntimeError('coordinates exceed declared grid resolution')
    lower=torch.as_tensor(aabb,device=coords.device,dtype=torch.float32)
    step=(lower[1]-lower[0])/grid
    vertices=(coords.float()+dual_vertices)*step+lower[0]
    key=lambda c:(c[...,0].long()*grid[1]+c[...,1].long())*grid[2]+c[...,2].long()
    ids,axis=torch.nonzero(intersected_flag,as_tuple=True)
    offsets=torch.tensor(OFFSETS,device=coords.device,dtype=coords.dtype)
    neighbors=coords[ids,None,:]+offsets[axis]
    bounds=((neighbors>=0)&(neighbors<grid)).all(-1)
    keys=key(coords)
    if method=='sorted':
        ordered,permutation=keys.sort()
        if bool((ordered[1:]==ordered[:-1]).any()):raise RuntimeError('native decoded coordinates contain duplicates')
        query=key(neighbors)
        locations=torch.searchsorted(ordered,query)
        safe=locations.clamp_max(max(len(ordered)-1,0))
        valid=bounds&(locations<len(ordered))&(ordered[safe]==query)
        indices=permutation[safe]
    elif method=='dense':
        table=torch.full((int(grid.prod()),),-1,device=coords.device,dtype=torch.int32)
        table[keys]=torch.arange(len(coords),device=coords.device,dtype=torch.int32)
        query=key(neighbors).clamp(0,len(table)-1)
        indices=table[query].long();valid=bounds&(indices>=0)
    else:raise ValueError(method)
    quads=indices[valid.all(-1)]
    if split_weight is None:raise ValueError('Shape decoder is expected to provide native split weights')
    weights=split_weight[quads]
    diagonal02=weights[:,0]*weights[:,2];diagonal13=weights[:,1]*weights[:,3]
    split1=quads[:,[0,1,2,0,2,3]];split2=quads[:,[0,1,3,3,1,2]]
    faces=torch.where(diagonal02>diagonal13,split1,split2).reshape(-1,3).int()
    return vertices,faces
