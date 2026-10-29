"""Diagnose serialization/order/numerical differences without editing assets."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .config import OUT,case,original,read,write,now,setup
from ...runtime import load_model,decode_native
from ...geometry import load_mesh,transform,observations,render


def main():
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0');root=original(5,42)
    saved=torch.load(root/'original_latent.pt',weights_only=False)
    decoder=load_model('shape_decoder',checkpoint=True,binding_root=OUT)
    with torch.no_grad():
        v,f=decode_native(decoder,saved['features'].cuda().float(),saved['coords'].cuda())
        bv,bf=load_mesh(root/'aligned_base.obj')
        matrix=torch.tensor(read(root/'fit.json')['matrix'],device='cuda:0',dtype=torch.float32)
        world=transform(v,matrix)
        _,k,w,_,_=observations(5)
        a=render(world,f,k,w,context);b=render(bv,bf,k,w,context)
        record=dict(time=now(),native_vertices=len(v),native_faces=len(f),
              obj_vertices=len(bv),obj_faces=len(bf),
              pointwise_max=float((world-bv).abs().max()) if world.shape==bv.shape else None,
              mask_difference_pixels=int((a['mask']!=b['mask']).sum()),
              common_depth_max=float((a['depth']-b['depth']).abs()[a['mask']&b['mask']].max()))
        if len(f)==len(bf):
            ids=torch.linspace(0,len(f)-1,min(50000,len(f)),device='cuda:0').long()
            record['same_face_index_triangle_position_max']=float((world[f[ids].long()]-bv[bf[ids].long()]).abs().max())
        points=world.cpu().numpy();points2=bv.cpu().numpy()
        ids=np.linspace(0,len(points)-1,min(len(points),20000),dtype=int)
        distance=cKDTree(points2).query(points[ids],workers=2)[0]
        record['native_to_obj_nn_max']=float(distance.max());record['native_to_obj_nn_mean']=float(distance.mean())
        write(OUT/'validation/baseline_replay.json',record);print(record,flush=True)


if __name__=='__main__':main()
