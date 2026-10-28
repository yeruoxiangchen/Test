from contextlib import contextmanager
import numpy as np
import torch

from visual_evidence_probe.experiments.local_repair import photometric
from visual_evidence_probe.experiments.local_repair.config import PARAMS as MATCH


@contextmanager
def configured(train,heldout=None):
    original=dict(MATCH)
    try:
        MATCH.update(train_views=list(train),min_support_views=1,uniqueness_gap=.01,uniqueness_radius=.01)
        if heldout is not None:MATCH['validation_view']=heldout
        yield
    finally:
        MATCH.clear();MATCH.update(original)


@torch.no_grad()
def propose(pred,gray,eroded,k,w,train):
    with configured(train):
        p=MATCH;offsets=pred['depth'].new_tensor(p['depth_offsets']);zero=len(offsets)//2
        result={name:[] for name in ('delta','valid','confidence')}
        for i in train:
            costs,depths,supports=[],[],[]
            for j in train:
                if i==j:continue
                volume,valid=photometric.pair_volume(pred,gray,eroded,k,w,i,j,offsets)
                costs.append(volume);depths.append(offsets[volume.argmin(0)]);supports.append(valid[zero])
            support=torch.stack(supports);count=support.sum(0)
            volume=torch.stack(costs);average=(volume*support[:,None]).sum(0)/count.clamp_min(1)
            best=average.argmin(0);best_cost=average.gather(0,best[None])[0];initial=average[zero]
            depths=torch.stack(depths)
            lo=torch.where(support,depths,torch.full_like(depths,1.)).min(0).values
            hi=torch.where(support,depths,torch.full_like(depths,-1.)).max(0).values
            far=(offsets[:,None,None]-offsets[best][None]).abs()>p['uniqueness_radius']+1.e-6
            alternative=torch.where(far,average,torch.full_like(average,2.)).min(0).values
            valid=(count>=1)&(hi-lo<=p['pair_depth_agreement'])&(initial-best_cost>=p['min_ncc_gain'])
            valid&=(best_cost<=p['max_ncc_cost'])&(alternative-best_cost>=p['uniqueness_gap'])
            valid&=(best>0)&(best<len(offsets)-1)&(best!=zero)
            confidence=((initial-best_cost).clamp_min(0)*(1-best_cost).clamp_min(0))*valid
            for name,value in dict(delta=offsets[best],valid=valid,confidence=confidence).items():result[name].append(value.cpu().numpy())
    return {name:np.stack(values) for name,values in result.items()}
