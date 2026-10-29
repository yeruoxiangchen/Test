import time
import torch

from .config import PARAMS
from .backend import decode
from .objective import loss
from ...runtime import release

class Guide:
    def __init__(self,decoder,flow,coords,mean,std,record,context):
        self.decoder,self.flow,self.coords=decoder,flow,coords
        self.mean,self.std,self.record,self.context=mean,std,record,context
        self.curve=[]

    def evaluate(self,z,origin=None,active=None):
        fields=decode(self.decoder,z*self.std+self.mean,self.coords)
        value,stats=loss(fields,self.active_record,self.context)
        prior=(z-origin)[active].square().mean() if origin is not None else z.sum()*0
        value=value+PARAMS['weights']['prior']*prior
        stats.update(prior=float(prior.detach()),loss=float(value.detach()))
        return value,stats

    def __call__(self,estimate,active,step,t):
        started=time.monotonic();active=torch.broadcast_to(active,estimate.shape)
        if len(self.record['points'])<PARAMS['min_queries']:
            self.curve.append(dict(step=step,time=t,skipped='insufficient fixed target queries'));return estimate
        if PARAMS['offload_flow']:
            self.flow.cpu();torch.cuda.empty_cache()
        try:
            self.active_record=dict(self.record)
            with torch.enable_grad():
                z=estimate.detach().requires_grad_(True);value,before=self.evaluate(z,estimate,active)
                if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite geometry objective')
                gradient=torch.autograd.grad(value*PARAMS['loss_scale'],z)[0]/PARAMS['loss_scale']
            if not bool(torch.isfinite(gradient).all()):raise RuntimeError('nonfinite clean-state geometry gradient')
            gradient=torch.where(active,gradient,torch.zeros_like(gradient))
            norm=float(gradient.norm());rms=gradient[active].square().mean().sqrt().clamp_min(1.e-10)
            direction=(gradient/rms).clamp(-3,3);del z,value,gradient
            release();chosen=estimate;after=before;used=0.
            for fraction in PARAMS['backtracking']:
                delta=(-PARAMS['guide_lr']*fraction*direction).clamp(-PARAMS['guide_cap'],PARAMS['guide_cap'])
                candidate=torch.where(active,estimate+delta,estimate)
                with torch.no_grad():_,stats=self.evaluate(candidate,estimate,active)
                if __import__('math').isfinite(stats['loss']) and stats['loss']<before['loss']-1.e-6:
                    chosen=candidate;after=stats;used=fraction;break
            self.curve.append(dict(step=step,time=t,gradient_norm=norm,accepted=used>0,step_fraction=used,
                before=before,after=after,clean_delta_max=float((chosen-estimate).abs().max()),seconds=time.monotonic()-started))
            return chosen.detach()
        finally:
            release()
            if PARAMS['offload_flow']:self.flow.cuda()
