import numpy as np
import torch

from .config import BASE,DIAGNOSTIC,PARAMS,source,case,read,write,save_tensor
from ...geometry import observations,transform,render,detached
from visual_evidence_probe.experiments.local_repair import photometric as photo
from visual_evidence_probe.experiments.dense_evidence.matching import configured


def load(index,seed,context):
    rec,k,w,images,masks=observations(index)
    path=DIAGNOSTIC/'objects'/f'{index:03d}'/str(seed)/'baseline_geometry.pt'
    geometry=torch.load(path,weights_only=False)
    old_matrix=torch.tensor(read(source(index,seed)/'fit.json')['matrix'],device='cuda:0',dtype=torch.float32)
    native=transform(geometry['vertices'].cuda(),torch.linalg.inv(old_matrix));faces=geometry['faces'].cuda()
    rgb,gray,observed,eroded=photo.prepare_images(images,masks,PARAMS['side'])
    evidence=read(source(index,seed)/'evidence.json');train=evidence['train_views'];heldout=evidence['heldout_views']
    baseline=detached(render(transform(native,old_matrix),faces,k,w,context))
    with configured(train,heldout[0]):supports=photo.validation_reference(baseline,gray,eroded,k,w)
    return dict(uid=rec['uid'],k=k,w=w,images=images,observed=observed,eroded=eroded,rgb=rgb,gray=gray,
                train=train,heldout=heldout,native=native,faces=faces,old_matrix=old_matrix,
                old_baseline=baseline,supports=supports)


def cache_plan(index,seed,matrix,trusted,roi,metadata):
    root=case(index,seed)
    save_tensor(root/'plan.pt',dict(matrix=matrix.cpu(),trusted={key:(value.cpu() if torch.is_tensor(value) else value)
                for key,value in trusted.items()},roi=roi))
    write(root/'plan.json',metadata)
