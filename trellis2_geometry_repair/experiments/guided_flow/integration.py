"""Euler correction through predicted clean state, with no Flow backward graph."""
import torch
from ..local_regeneration.sampling import forward_noise,schedule

def clean_correction(proposal,before,after,t,previous,sigma_min):
    # x0 = (1-sigma_min)*xt - a(t)*v. Correct v, then perform the SAME Euler step.
    coefficient=(t-previous)/(sigma_min+(1-sigma_min)*t)
    return proposal+coefficient*(after-before)

@torch.no_grad()
def integrate(sampler,flow,clean,active,noise,condition,kwargs,guide=None,steps=12,rescale=3.,guide_steps=(9,10,11),callback=None):
    active=torch.broadcast_to(active.bool(),clean.shape)
    x=forward_noise(clean,noise,1.,sampler.sigma_min);times=schedule(1.,steps,rescale)
    from ...runtime import sparse
    coords=kwargs.pop('_coords')
    for step,(t,previous) in enumerate(zip(times[:-1],times[1:])):
        value=sampler.sample_once(flow,sparse(x,coords),t,previous,cond=condition,**kwargs)
        proposal=value.pred_x_prev.feats
        if guide is not None and step in guide_steps:
            estimate=torch.where(active,value.pred_x_0.feats,clean)
            corrected=guide(estimate.detach(),active,step,t)
            if not torch.equal(corrected[~active],estimate[~active]):raise RuntimeError('guide changed exterior clean estimate')
            proposal=clean_correction(proposal,estimate,corrected,t,previous,sampler.sigma_min)
        x=torch.where(active,proposal,forward_noise(clean,noise,previous,sampler.sigma_min))
        if not bool(torch.isfinite(x).all()):raise RuntimeError('nonfinite guided Flow state')
        if callback:callback(step+1,previous)
    return torch.where(active,x,clean)
