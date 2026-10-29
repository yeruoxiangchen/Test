"""Continuous surface/routing constraints plus native rendering and preservation."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .config import PARAMS
from .backend import vertices
from ..local_regeneration.fields import keys,lookup,to_mesh
from ..local_regeneration.evidence import mask_cost
from ...geometry import transform,render

def match(source,query):
    if not len(source):return torch.zeros(len(query),device=query.device,dtype=torch.long),torch.zeros(len(query),device=query.device,dtype=torch.bool)
    values,order=keys(source).sort();q=keys(query);position=torch.searchsorted(values,q)
    valid=(position<len(values))&((query>=0)&(query<512)).all(1);safe=position.clamp_max(len(values)-1);valid &= values[safe]==q
    return order[safe],valid

def log_surface(raw):
    # Probability of at least one oriented edge crossing. NOT voxel occupancy.
    positive=F.logsigmoid(raw[:,3:6]);negative=F.logsigmoid(-raw[:,3:6])
    return torch.logsumexp(torch.stack((positive[:,0],negative[:,0]+positive[:,1],negative[:,:2].sum(1)+positive[:,2]),1),1)

def route_values(sub,points):
    grid=sub['grid'];child=torch.floor((points+.5)*(grid*2)).long()
    valid=((child>=0)&(child<grid*2)).all(1);parent=child//2
    idx,present=match(sub['coords'],parent);valid &= present
    channel=(child[:,0]%2)+2*(child[:,1]%2)+4*(child[:,2]%2)
    cost=sub['raw'].new_zeros(len(points),dtype=torch.float32)
    if bool(valid.any()):cost=cost.index_copy(0,torch.nonzero(valid).flatten(),F.softplus(-sub['raw'][idx[valid],channel[valid]].float()))
    return cost,valid

def missing_cost(cost,valid,margin=1.,floor=1.):
    fallback=cost[valid].detach().max().clamp_min(floor)+margin if bool(valid.any()) else cost.new_tensor(floor+margin)
    return torch.where(valid,cost.detach()+margin,fallback)

def routing_loss(subs,points,penalties=None):
    terms=[];coverage=[]
    for level,sub in enumerate(subs):
        cost,valid=route_values(sub,points)
        frozen=missing_cost(cost,valid) if penalties is None else penalties[level]
        full=torch.where(valid,cost,frozen)
        if len(points):terms.append(full.mean()+sub['raw'].float().sum()*0)
        coverage.append(dict(grid=sub['grid'],queries=len(points),reachable_parent=int(valid.sum()),
                             fixed_denominator=len(points),missing_penalty_max=float(frozen.max()) if len(frozen) else 0.))
    zero=subs[0]['raw'].float().sum()*0 if subs else points.sum()*0
    return (torch.stack(terms).mean() if terms else zero),coverage

def surface_costs(fields,points):
    zero=fields['raw'].sum()*0
    if not len(points) or not len(fields['coords']):return zero.expand(len(points)),torch.zeros(len(points),device=points.device,dtype=torch.bool)
    with torch.no_grad():
        v=vertices(fields);tree=cKDTree(v.detach().cpu().numpy())
        _,ids=tree.query(points.detach().cpu().numpy(),k=min(8,len(v)),workers=2)
        if ids.ndim==1:ids=ids[:,None]
        ids=torch.as_tensor(ids,device=points.device)
    v=vertices(fields)[ids];distance=torch.sqrt(((v-points[:,None])/PARAMS['point_scale']).square().sum(2)+1.e-4)-.01
    logp=log_surface(fields['raw'])[ids]
    cost=-torch.logsumexp(logp-distance,dim=1)+math.log(ids.shape[1])
    return cost,torch.ones(len(points),device=points.device,dtype=torch.bool)

def surface_loss(fields,points,weights,penalties=None):
    cost,valid=surface_costs(fields,points)
    frozen=missing_cost(cost,valid) if penalties is None else penalties
    cost=torch.where(valid,cost,frozen)
    return (cost*weights).sum()/weights.sum().clamp_min(1.e-8),dict(queries=len(points),field_cells=len(fields['coords']))

def free_loss(fields,free_points,surface_points):
    cell=torch.floor((free_points+.5)*512).long();idx,present=match(fields['coords'],cell)
    if len(surface_points):
        tree=cKDTree(surface_points.detach().cpu().numpy());distance=tree.query(free_points.detach().cpu().numpy(),workers=2)[0]
        present &= torch.as_tensor(distance>3/512,device=cell.device)
    if not bool(present.any()):return fields['raw'].sum()*0,0
    penalty=F.softplus(fields['raw'][idx[present],3:6].float()).mean(1).sum()
    return penalty/max(len(free_points),1),int(present.sum())

def protection_costs(fields,reference):
    values=[];validity=[];idx,present=match(fields['coords'],reference['coords'])
    cost=fields['raw'].new_zeros(len(reference['coords']),dtype=torch.float32)
    if bool(present.any()):
        a=fields['raw'][idx[present]].float();b=reference['raw'][present].float()
        cost=cost.index_copy(0,torch.nonzero(present).flatten(),(a[:,:3].sigmoid()-b[:,:3].sigmoid()).abs().mean(1)+F.binary_cross_entropy_with_logits(a[:,3:6],b[:,3:6].sigmoid(),reduction='none').mean(1))
    values.append(cost);validity.append(present)
    for sub,ref in zip(fields['subs'],reference['subs']):
        idx,present=match(sub['coords'],ref['coords'])
        cost=fields['raw'].new_zeros(len(ref['coords']),dtype=torch.float32)
        if bool(present.any()):cost=cost.index_copy(0,torch.nonzero(present).flatten(),F.binary_cross_entropy_with_logits(sub['raw'][idx[present]].float(),ref['raw'][present].float().sigmoid(),reduction='none').mean(1))
        values.append(cost);validity.append(present)
    return values,validity

def field_protect(fields,reference,penalties=None):
    costs,validity=protection_costs(fields,reference);terms=[]
    for level,(cost,valid) in enumerate(zip(costs,validity)):
        if not len(cost):continue
        frozen=missing_cost(cost,valid) if penalties is None else penalties[level]
        terms.append(torch.where(valid,cost,frozen).mean())
    return torch.stack(terms).mean() if terms else fields['raw'].sum()*0

def fixed_depth_cost(pred,trusted,train,missing):
    valid=trusted['valid'][train];residual=(pred['depth'][train]-trusted['depth'][train]).abs()
    value=torch.where(pred['mask'][train],residual,missing[train])
    cost=torch.sqrt((value/.03).square()+1.e-4)-.01
    return (cost*valid).sum()/valid.sum().clamp_min(1)

def loss(fields,record,context):
    zero=fields['raw'].sum()*0
    local=lookup(record['envelope'],fields['coords'],16).bool()
    subset=dict(coords=fields['coords'][local],raw=fields['raw'][local])
    if len(fields['coords']):
        v,f=to_mesh(fields)
    else:v,f=None,None
    pred=render(transform(v,record['matrix']),f,record['k'],record['w'],context) if f is not None and len(f) else None
    if pred is None:
        shape=record['baseline']['depth'].shape
        pred=dict(depth=zero.expand(shape),mask=torch.zeros(shape,device=zero.device,dtype=torch.bool),softmask=zero.expand(shape))
    if '_fixed_missing' not in record:
        point_cost,point_valid=surface_costs(subset,record['points'])
        refs=dict(surface=missing_cost(point_cost,point_valid),routing=[])
        for sub in fields['subs']:
            cost,valid=route_values(sub,record['points']);refs['routing'].append(missing_cost(cost,valid))
        costs,validity=protection_costs(fields,record['field_reference'])
        refs['field']=[missing_cost(c,v) for c,v in zip(costs,validity)]
        old=(record['baseline']['depth']-record['trusted']['depth']).abs()
        current=torch.where(pred['mask'],(pred['depth']-record['trusted']['depth']).abs(),old)
        refs['depth']=torch.maximum(old,current).detach().clamp_min(.1)+.05
        current=torch.where(pred['mask'],(pred['depth']-record['baseline']['depth']).abs(),torch.full_like(old,.1))
        refs['protect']=current.detach().clamp_min(.1)+.05
        record['_fixed_missing']=refs
    refs=record['_fixed_missing']
    point,point_stats=surface_loss(subset,record['points'],record['point_weights'],refs['surface'])
    route,coverage=routing_loss(fields['subs'],record['points'],refs['routing'])
    free,free_count=free_loss(subset,record['free_points'],record['points'])
    protected=field_protect(fields,record['field_reference'],refs['field'])
    values=dict(surface=point,subdivision=route,free=free,field_protect=protected)
    values['depth']=fixed_depth_cost(pred,record['trusted'],record['train'],refs['depth'])
    values['silhouette']=mask_cost(pred,record['observed'],record['train'])
    stable=record['protected_pixels']
    error=(pred['depth']-record['baseline']['depth']).abs()
    error=torch.where(pred['mask'],error,refs['protect'])
    depth=(torch.sqrt((error/.03).square()+.0001)-.01)*stable
    values['protect']=depth.sum()/stable.sum().clamp_min(1)
    values['protect']=values['protect']+((pred['softmask']-record['baseline']['softmask']).abs()*stable).sum()/stable.sum().clamp_min(1)
    total=sum(PARAMS['weights'][name]*value for name,value in values.items())
    return total,dict(**{name:float(value.detach()) for name,value in values.items()},
                      loss=float(total.detach()),surface_queries=point_stats['queries'],
                      surface_cells=point_stats['field_cells'],free_cells=free_count,routing=coverage)
