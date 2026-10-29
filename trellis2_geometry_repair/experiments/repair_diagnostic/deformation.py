"""Input-defined smooth graph deformation; exterior vertices stay exact."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .config import PARAMS


def farthest(points,count):
    # Deterministic candidate thinning avoids a huge all-vertices FPS matrix.
    if len(points)>4096:points=points[np.linspace(0,len(points)-1,4096,dtype=int)]
    if len(points)<=count:return points
    selected=[0];dist=np.full(len(points),np.inf)
    for _ in range(1,count):
        dist=np.minimum(dist,((points-points[selected[-1]])**2).sum(1))
        selected.append(int(dist.argmax()))
    return points[selected]


class Graph(torch.nn.Module):
    def __init__(self,vertices,faces,baseline,region,train):
        super().__init__();self.register_buffer('vertices',vertices.detach())
        ids=baseline['face_id'][train][(region[train]>.5)&baseline['mask'][train]].unique()
        if not len(ids):raise RuntimeError('no input-supported mesh edit vertices')
        roi_ids=faces[ids.long()].long().reshape(-1).unique()
        points=vertices[roi_ids].detach().cpu().numpy();anchors=farthest(points,PARAMS['mesh_anchors'])
        all_points=vertices.detach().cpu().numpy()
        distance=cKDTree(points).query(all_points,workers=2)[0]
        editable=np.flatnonzero(distance<PARAMS['mesh_taper'])
        d,neighbors=cKDTree(anchors).query(all_points[editable],k=min(PARAMS['mesh_neighbors'],len(anchors)),workers=2)
        if d.ndim==1:d,neighbors=d[:,None],neighbors[:,None]
        weights=1/np.maximum(d,1.e-4)**2;weights/=weights.sum(1,keepdims=True)
        taper=np.clip(1-distance[editable]/PARAMS['mesh_taper'],0,1)
        taper=taper*taper*(3-2*taper)
        device=vertices.device
        self.register_buffer('editable',torch.tensor(editable,device=device))
        self.register_buffer('neighbors',torch.tensor(neighbors,device=device))
        self.register_buffer('weights',torch.tensor(weights,device=device,dtype=torch.float32))
        self.register_buffer('taper',torch.tensor(taper,device=device,dtype=torch.float32))
        self.delta=torch.nn.Parameter(vertices.new_zeros((len(anchors),3)))
        _,nn=cKDTree(anchors).query(anchors,k=min(5,len(anchors)))
        if nn.ndim==1:nn=nn[:,None]
        self.register_buffer('adjacent',torch.tensor(nn[:,1:],device=device))
        self.info=dict(anchors=len(anchors),editable_vertices=len(editable),total_vertices=len(vertices),
                       editable_fraction=len(editable)/len(vertices),outside_vertices_exact=True,
                       input_defined=True,max_translation_per_axis=PARAMS['mesh_cap'])

    def forward(self):
        displacement=PARAMS['mesh_cap']*torch.tanh(self.delta)
        local=(displacement[self.neighbors]*self.weights[...,None]).sum(1)*self.taper[:,None]
        return self.vertices.index_add(0,self.editable,local)

    def regularizer(self):
        value=torch.tanh(self.delta)
        smooth=(value[:,None]-value[self.adjacent]).square().mean() if self.adjacent.numel() else value.sum()*0
        return PARAMS['mesh_prior']*value.square().mean()+PARAMS['mesh_smooth']*smooth
