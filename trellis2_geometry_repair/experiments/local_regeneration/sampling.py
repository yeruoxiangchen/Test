"""Masked Euler integration with the upstream time/noise convention."""
import numpy as np
import torch


def forward_noise(clean,noise,t,sigma_min=1.e-5):
    return (1-t)*clean+(sigma_min+(1-sigma_min)*t)*noise


def schedule(start,steps,rescale):
    if not 0<=start<=1 or steps<1 or rescale<=0:raise ValueError('invalid flow schedule')
    u=np.linspace(1.,0.,steps+1)
    values=(start*rescale*u/(1+(rescale-1)*u)).tolist()
    values[0]=float(start);values[-1]=0.
    return values


@torch.no_grad()
def masked_euler(sampler,model,clean,mask,noise,start,steps,rescale,condition,
                 wrap=lambda x:x,unwrap=lambda x:x,kwargs=None,callback=None):
    if clean.shape!=noise.shape:raise ValueError('noise/clean shape mismatch')
    active=torch.broadcast_to(mask.bool(),clean.shape)
    if start==0 or not bool(active.any()):return clean.clone()
    x=forward_noise(clean,noise,start,sampler.sigma_min)
    times=schedule(start,steps,rescale)
    for n,(t,previous) in enumerate(zip(times[:-1],times[1:])):
        # Fresh SparseTensor caches on each model evaluation.
        value=sampler.sample_once(model,wrap(x),t,previous,cond=condition,**(kwargs or {}))
        proposal=unwrap(value.pred_x_prev)
        outside=forward_noise(clean,noise,previous,sampler.sigma_min)
        x=torch.where(active,proposal,outside)
        if not bool(torch.isfinite(x).all()):raise RuntimeError('nonfinite local Flow state')
        if callback:callback(n+1,previous)
    # The sigma_min terminal noise must not perturb preserved clean entries.
    return torch.where(active,x,clean)


def coordinate_keys(coords,grid=32):
    c=coords.long()
    return ((c[:,0]*grid+c[:,1])*grid+c[:,2])*grid+c[:,3]


def join_original(old_coords,new_coords):
    old=coordinate_keys(old_coords);new=coordinate_keys(new_coords)
    values,order=old.sort();position=torch.searchsorted(values,new)
    valid=position<len(values);safe=position.clamp_max(max(len(values)-1,0))
    valid &= values[safe]==new
    indices=order[safe]
    return indices,valid


def restore_exterior(denormalized,original_features,original_coords,new_coords,editable):
    indices,present=join_original(original_coords,new_coords)
    outside=~editable
    if not bool(present[outside].all()):raise RuntimeError('new exterior SS coordinate')
    result=denormalized.clone();result[outside]=original_features[indices[outside]]
    if not torch.equal(result[outside],original_features[indices[outside]]):raise RuntimeError('exterior feature drift')
    return result


def editable_volume_lookup(old_coords,new_coords,editable):
    # Coordinate matching utility, not a geometric locality assertion.
    new_keys=coordinate_keys(new_coords);old_keys=coordinate_keys(old_coords)
    values,order=new_keys.sort();pos=torch.searchsorted(values,old_keys)
    valid=pos<len(values);safe=pos.clamp_max(max(len(values)-1,0));valid &= values[safe]==old_keys
    output=torch.zeros(len(old_coords),device=old_coords.device,dtype=torch.bool)
    output[valid]=editable[order[safe[valid]]]
    return output
