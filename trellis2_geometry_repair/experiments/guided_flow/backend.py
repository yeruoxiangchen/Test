"""Native decoder before hard subdivision/intersection decisions, retaining gradients."""
import torch
from ...runtime import sparse
from ..local_regeneration.fields import lookup,to_mesh

def decode(decoder,features,coords):
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    if decoder.resolution!=512:raise RuntimeError('native512 required')
    h,subs=SparseUnetVaeDecoder.forward(decoder,sparse(features,coords),return_subs=True)
    return dict(coords=h.coords[:,1:],raw=h.feats,
                subs=[dict(coords=s.coords[:,1:],raw=s.feats,grid=32*2**i) for i,s in enumerate(subs)])

def vertices(fields):
    return (fields['coords'].float()+2*fields['raw'][:,:3].sigmoid()-.5)/512-.5

def take_even(indices,limit):
    if len(indices)<=limit:return indices
    return indices[torch.linspace(0,len(indices)-1,limit,device=indices.device).long()]

@torch.no_grad()
def protection(fields,envelope,field_limit=4096,sub_limit=512):
    ids=take_even(torch.nonzero(~lookup(envelope,fields['coords'],16).bool()).flatten(),field_limit)
    result=dict(coords=fields['coords'][ids].detach(),raw=fields['raw'][ids].detach(),subs=[])
    for sub in fields['subs']:
        ids=take_even(torch.nonzero(~lookup(envelope,sub['coords'],sub['grid']//32).bool()).flatten(),sub_limit)
        result['subs'].append(dict(coords=sub['coords'][ids].detach(),raw=sub['raw'][ids].detach(),grid=sub['grid']))
    return result
