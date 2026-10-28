"""Compare supported sparse-convolution kernels without changing weights."""
import time
import numpy as np
import torch

from .config import OUT,folder,setup,write,now
from .runtime import load_model,decode_native,release
from .geometry import observations,render


def main():
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    import flex_gemm.ops.spconv as spconv
    from trellis2.modules.sparse.conv import config as conv_config
    import nvdiffrast.torch as dr
    saved=torch.load(folder(5,42)/'original_latent.pt',weights_only=False)
    features=saved['features'].cuda();coords=saved['coords'].cuda()
    _,k,w,_,_=observations(5);context=dr.RasterizeCudaContext(device='cuda:0')
    decoder=load_model('shape_decoder',checkpoint=True);records=[]
    for mode in ('masked_implicit_gemm_splitk','masked_implicit_gemm','implicit_gemm'):
        conv_config.FLEX_GEMM_ALGO=mode;spconv.set_algorithm(mode);reference=None;rows=[]
        for iteration,grad in enumerate((False,False,True,True)):
            x=features.detach().clone().requires_grad_(grad);start=time.monotonic()
            with torch.set_grad_enabled(grad):v,f=decode_native(decoder,x,coords)
            if spconv.ALGORITHM!=mode:raise RuntimeError('kernel selection was overridden')
            with torch.no_grad():pred=render(v,f,k,w,context,128)
            current=dict(vertices=v.detach().cpu().numpy(),faces=f.cpu().numpy(),depth=pred['depth'].cpu().numpy(),mask=pred['mask'].cpu().numpy())
            unordered=np.sort(current['faces'],axis=1)
            current['face_set']=unordered[np.lexsort((unordered[:,2],unordered[:,1],unordered[:,0]))]
            if reference is None:reference=current
            same_shape=current['vertices'].shape==reference['vertices'].shape
            depth_error=np.abs(current['depth']-reference['depth'])
            union=current['mask']|reference['mask']
            row=dict(iteration=iteration,gradient_enabled=grad,vertices=len(v),faces=len(f),
                     same_vertex_shape=same_shape,vertex_max_abs=float(np.max(np.abs(current['vertices']-reference['vertices']))) if same_shape else None,
                     faces_equal=bool(np.array_equal(current['faces'],reference['faces'])),
                     triangle_sets_equal=bool(np.array_equal(current['face_set'],reference['face_set'])),
                     mask_mismatch_pixels=int((current['mask']^reference['mask']).sum()),
                     depth_max_abs=float(depth_error.max()),depth_mean=float(depth_error[union].mean()),seconds=time.monotonic()-start)
            rows.append(row);print(dict(mode=mode,**row),flush=True)
            del v,f,pred,x;release()
        records.append(dict(mode=mode,rows=rows))
        write(OUT/'validation/native_replay_512.json',dict(time=now(),records=records,gt_used=False,decoder_resolution=512))


if __name__=='__main__':main()
