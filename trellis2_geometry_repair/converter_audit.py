import torch
import numpy as np

from .config import OUT,folder,setup,write,now
from .runtime import load_model,decode_native,release
from .fdg import extract


def canonical(faces):
    array=np.sort(faces.detach().cpu().numpy(),axis=1)
    return array[np.lexsort((array[:,2],array[:,1],array[:,0]))]


def main():
    setup();torch.set_num_threads(2)
    from trellis2.models.sc_vaes import fdg_vae
    original=fdg_vae.flexible_dual_grid_to_mesh;captured=[]
    def capture(coords,vertices,flags,weights,**kwargs):
        if not captured:captured.append((coords.detach().clone(),vertices.detach().clone(),flags.detach().clone(),weights.detach().clone(),kwargs))
        return original(coords,vertices,flags,weights,**kwargs)
    fdg_vae.flexible_dual_grid_to_mesh=capture
    saved=torch.load(folder(5,42)/'original_latent.pt',weights_only=False)
    decoder=load_model('shape_decoder')
    with torch.no_grad():
        decode_native(decoder,saved['features'].cuda(),saved['coords'].cuda())
    fdg_vae.flexible_dual_grid_to_mesh=original
    coords,vertices,flags,weights,kwargs=captured[0];del decoder;release()
    records=[];reference=None
    with torch.no_grad():
        for i in range(4):
            v,f=original(coords,vertices,flags,weights,**kwargs)
            actual=canonical(f)
            if reference is None:reference=actual
            row=dict(iteration=i,vertices=len(v),faces=len(f),triangle_set_equal_to_first=bool(np.array_equal(actual,reference)))
            records.append(row);print(dict(original_converter=row),flush=True)
        sorted_v,sorted_f=extract(coords,vertices,flags,weights,kwargs['aabb'],kwargs['grid_size'],'sorted')
        dense_v,dense_f=extract(coords,vertices,flags,weights,kwargs['aabb'],kwargs['grid_size'],'dense')
        agree=bool(torch.equal(sorted_f,dense_f) and torch.equal(sorted_v,dense_v))
        second_v,second_f=extract(coords,vertices,flags,weights,kwargs['aabb'],kwargs['grid_size'],'sorted')
        repeat=bool(torch.equal(sorted_f,second_f) and torch.equal(sorted_v,second_v))
    result=dict(time=now(),fixed_raw_fgd_input=True,original_replays=records,
                deterministic_sorted_and_dense_agree=agree,deterministic_repeat_exact=repeat,
                vertices=len(sorted_v),faces=len(sorted_f),
                original_first_matches_deterministic=bool(np.array_equal(reference,canonical(sorted_f))))
    write(OUT/'validation/converter_audit.json',result);print(result,flush=True)
    if not agree or not repeat:raise RuntimeError('deterministic lookup validation failed')


if __name__=='__main__':main()
