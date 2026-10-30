"""Explicit masked flow sampling in normalized native T2 latent coordinates."""
from __future__ import annotations
import torch
from .model import features, replace, conditional_sample
from .vendor_pose.constants import SIGMA_MIN


def diffuse(clean, noise, t):
    return replace(clean, (1-t)*features(clean) + (SIGMA_MIN+(1-SIGMA_MIN)*t)*features(noise))


def target_velocity(clean, noise):
    return (1-SIGMA_MIN)*features(noise) - features(clean)


def bridge_state(clean,bad,noise,t,sigma=.1):
    """Train and sample the same damaged-to-clean interpolant.

    y(t)=(1-t)*clean+t*bad. Noise is a known augmentation of the network
    input, not an untrained perturbation left in the repaired endpoint.
    """
    return replace(clean,(1-t)*features(clean)+t*features(bad)+sigma*t*features(noise))


@torch.no_grad()
def sample_bridge(model,sample,intervention,*,seed,steps=12,sigma=.1):
    ref=intervention['bad']
    gen=torch.Generator(device=ref.device).manual_seed(int(seed))
    noise=torch.randn(features(ref).shape,device=ref.device,generator=gen)
    y=replace(ref,features(ref).clone())
    context=sample['dino_features'][:1].to(ref.device)
    views=torch.arange(len(sample['dino_features']))
    cond=conditional_sample(sample,intervention,True)
    cached=model._pose_condition(y,cond,views)
    times=torch.linspace(1,0,steps+1,device=ref.device)
    for t,tn in zip(times[:-1],times[1:]):
        x=replace(y,features(y)+sigma*t*noise)
        native_t=t.reshape(1)*650
        stock=model.stock_prediction(x,native_t,context)
        adapted,_=model.adapted_prediction(x,native_t,context,cond,view_indices=views,
            precomputed_condition=cached,stock_velocity=stock,compare_stock=False)
        velocity=features(adapted).float()-features(stock).float()
        y=replace(y,features(y)+(tn-t)*velocity)
        if not torch.is_tensor(y):
            y=replace(y,torch.where(intervention['mask'][:,None],features(y),features(ref)))
    if not torch.isfinite(features(y)).all():
        raise ValueError('Nonfinite bridge endpoint')
    return y


def weighted_flow_loss(prediction, target, mask, stage, arm):
    squared = (features(prediction).float() - target.float()).square()
    if stage == 'ss':
        m = mask.expand_as(squared)
    else:
        m = mask[:,None].expand_as(squared)
    inside = (squared*m).sum()/m.sum().clamp_min(1)
    outside = (squared*(~m)).sum()/(~m).sum().clamp_min(1)
    if arm == 'full':
        loss = squared.mean()
    else:
        # Normalize each region separately, so tiny errors cannot disappear
        # in the much larger correctly reconstructed region.
        loss = inside + (.25 if stage == 'ss' else .1)*outside
    return loss, inside, outside


@torch.no_grad()
def sample_flow(model, sample, intervention, *, seed, steps=12, strength=.65,
                cfg=1., arm='repair', stock=False):
    ref = intervention['bad']
    ref_feats = features(ref)
    gen = torch.Generator(device=ref_feats.device).manual_seed(int(seed))
    noise = replace(ref, torch.randn(ref_feats.shape, device=ref_feats.device,
                                    dtype=torch.float32, generator=gen))
    strength = 1. if arm == 'full' else float(strength)
    x = diffuse(ref, noise, strength)
    context = sample['dino_features'][:1].to(ref_feats.device)
    views = torch.arange(len(sample['dino_features']), device='cpu')
    positive = conditional_sample(sample, intervention, True)
    negative = conditional_sample(sample, intervention, False)
    pos_cond = None if stock else model._pose_condition(x, positive, views)
    neg_cond = None if stock or cfg == 1 else model._pose_condition(x, negative, views)
    # Geometry state is retained in both CFG branches. Only image evidence
    # is dropped in the negative branch, matching training.
    times = torch.linspace(strength, 0, steps+1, device=ref_feats.device)
    for t, tn in zip(times[:-1], times[1:]):
        timestep = t.reshape(1)*1000
        if stock:
            velocity = model.stock_prediction(x, timestep, context)
        else:
            velocity,_ = model.adapted_prediction(x,timestep,context,positive,
                view_indices=views,precomputed_condition=pos_cond,compare_stock=False)
            if cfg != 1:
                uncond,_ = model.adapted_prediction(x,timestep,torch.zeros_like(context),negative,
                    view_indices=views,precomputed_condition=neg_cond,compare_stock=False)
                velocity = replace(velocity, features(uncond)+cfg*(features(velocity)-features(uncond)))
        x = replace(x, features(x)+(tn-t)*features(velocity))
        if not torch.is_tensor(x) and arm == 'repair':
            keep = diffuse(ref,noise,tn)
            mask = intervention['mask'][:,None]
            x = replace(x, torch.where(mask,features(x),features(keep)))
    if not torch.is_tensor(x) and arm == 'repair':
        # Exact saved-feature preservation, rather than sigma_min residual.
        x = replace(x, torch.where(intervention['mask'][:,None],features(x),ref_feats))
        assert torch.equal(features(x)[~intervention['mask']],ref_feats[~intervention['mask']])
    if not torch.isfinite(features(x)).all():
        raise ValueError('Sampler returned nonfinite latent')
    return x
