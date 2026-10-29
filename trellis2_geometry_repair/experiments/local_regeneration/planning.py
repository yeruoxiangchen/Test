import argparse
import time
import numpy as np
import torch

from .config import OUT,PARAMS,case,source,read,write,setup,lock_case,save_tensor,emit,now
from .data import load,cache_plan
from .evidence import reliable_targets,volume
from .alignment import refine
from ...geometry import transform,render,detached


def one(index,seed,context):
    root=case(index,seed)
    if (root/'plan.json').exists():return
    started=time.monotonic()
    data=load(index,seed,context)
    trusted=reliable_targets(index,seed,data,data['old_baseline'])
    emit(index,seed,'global_registration',verified_pixels=int(trusted['valid'].sum()))
    matrix,alignment=refine(data,trusted,context)
    with torch.no_grad():
        baseline=detached(render(transform(data['native'],matrix),data['faces'],data['k'],data['w'],context))
        trusted=reliable_targets(index,seed,data,baseline)
        saved=torch.load(source(index,seed)/'original_latent.pt',weights_only=False)
        roi=volume(matrix,data,baseline,trusted,saved['coords'].cuda())
    metadata=dict(time=now(),index=index,seed=seed,uid=data['uid'],gt_used=False,alignment=alignment,
            trusted_pixels=int(trusted['valid'].sum()),da3_verified_pixels=trusted['da3_verified_pixels'],
            ncc_pixels=trusted['ncc_pixels'],core_cells=int(roi['core'].sum()),
            envelope_cells=int(roi['envelope'].sum()),active_fraction=roi['active_fraction'],
            verified_error_pixels=roi['verified_pixels'],extra_mask_pixels=roi['extra_pixels'],
            needs_ss=roi['needs_ss'],has_evidence=roi['has_evidence'],
            unsupported_positive_cells=roi['unsupported_positive_cells'],
            verified_anchor_cells=roi['anchor_cells'],
            scope='regional' if roi['active_fraction']>.6 else 'local',seconds=time.monotonic()-started)
    cache_plan(index,seed,matrix,trusted,roi,metadata)
    save_tensor(root/'registered_render.pt',{key:value.cpu() for key,value in baseline.items()})
    emit(index,seed,'plan_complete',core_cells=metadata['core_cells'],needs_ss=roi['needs_ss'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.7)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed):one(a.index,a.seed,dr.RasterizeCudaContext(device='cuda:0'))


if __name__=='__main__':main()
