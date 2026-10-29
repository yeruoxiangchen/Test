"""Original native FDG decoding and explicitly labeled local field composition."""
import torch
import torch.nn.functional as F

from ...runtime import sparse


def keys(coords,grid=512):
    c=coords.long();return (c[:,0]*grid+c[:,1])*grid+c[:,2]


def lookup(volume,coords,divisor=1):
    c=torch.div(coords.long(),divisor,rounding_mode='floor')
    valid=((c>=0)&(c<volume.shape[0])).all(1);c=c.clamp(0,volume.shape[0]-1)
    return volume[c[:,0],c[:,1],c[:,2]]*valid


@torch.no_grad()
def decode_fields(decoder,features,coords):
    captured={}
    def capture(module,args,output):captured['coords']=output.coords[:,1:].detach();captured['raw']=output.feats.detach()
    handle=decoder.output_layer.register_forward_hook(capture)
    try:mesh=decoder(sparse(features,coords))[0]
    finally:handle.remove()
    if decoder.resolution!=512 or not captured:raise RuntimeError('native512 field capture required')
    return mesh.vertices,mesh.faces,captured


def to_mesh(fields,margin=.5):
    from o_voxel.convert import flexible_dual_grid_to_mesh
    raw=fields['raw'];offsets=(1+2*margin)*torch.sigmoid(raw[:,:3])-margin
    return flexible_dual_grid_to_mesh(fields['coords'].int(),offsets,raw[:,3:6]>0,F.softplus(raw[:,6:7]),
          aabb=[[-.5,-.5,-.5],[.5,.5,.5]],grid_size=512,train=False)


def compose(base,candidate,alpha):
    """Outside raw fields remain exact; this is not decoder(one latent)."""
    if not bool((alpha>0).any()):
        return dict(coords=base['coords'].clone(),raw=base['raw'].clone()),dict(exterior_fields_exact=True,
          baseline_exterior_cells=len(base['coords']),native_candidate_cells=len(candidate['coords']),
          composite_cells=len(base['coords']),output='identity native FDG; not a changed latent decode')
    bkeys=keys(base['coords']);ckeys=keys(candidate['coords'])
    sorted_base,border=bkeys.sort();where=torch.searchsorted(sorted_base,ckeys)
    present=where<len(sorted_base);safe=where.clamp_max(len(sorted_base)-1)
    present &= sorted_base[safe]==ckeys;bindex=border[safe]
    ca=lookup(alpha,candidate['coords'],16)
    ba=lookup(alpha,base['coords'],16)
    # Preserve all baseline exterior/buffer cells. Interior support follows the
    # new native decoder. New cells enter only where the blend is mostly new.
    keep_base=ba<1-1.e-7
    use_candidate=(ca>=1-1.e-7)|((ca>.5)&~present)
    overlap=present&(ca>0)&(ca<1-1.e-7)
    updated=base['raw'].clone()
    if bool(overlap.any()):
        index=bindex[overlap];a=ca[overlap,None]
        updated[index]=torch.lerp(base['raw'][index],candidate['raw'][overlap],a)
    coords=torch.cat((base['coords'][keep_base],candidate['coords'][use_candidate]))
    raw=torch.cat((updated[keep_base],candidate['raw'][use_candidate]))
    order=keys(coords).argsort();coords,raw=coords[order],raw[order]
    if bool((keys(coords)[1:]==keys(coords)[:-1]).any()):raise RuntimeError('duplicate composite fine cells')
    # Verify exterior support and seven channels by spatial key, not row ID.
    outside=ba==0;values=keys(coords);pos=torch.searchsorted(values,bkeys[outside])
    if not torch.equal(values[pos],bkeys[outside]) or not torch.equal(raw[pos],base['raw'][outside]):
        raise RuntimeError('FDG exterior changed')
    return dict(coords=coords,raw=raw),dict(exterior_fields_exact=True,baseline_exterior_cells=int(outside.sum()),
               native_candidate_cells=len(ckeys),composite_cells=len(coords),
               output='native FDG local composite; not direct decode of saved latent')
