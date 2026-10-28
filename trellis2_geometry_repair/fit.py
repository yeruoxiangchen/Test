import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import PARAMS,folder,write,setup,snapshot,emit,now
from .geometry import observations,load_mesh,render,cube_rotations,similarity
from .runtime import save_mesh


def one(index,seed,context):
    root=folder(index,seed)
    if (root/'fit.json').exists():return
    start=time.monotonic();_,k,w,_,masks=observations(index)
    vertices,faces=load_mesh(root/'native_original.obj')
    target=F.interpolate(torch.tensor(masks,device='cuda:0',dtype=torch.float32)[:,None],size=(96,96),mode='area')[:,0]
    rotations=torch.tensor(cube_rotations(),device='cuda:0');ranking=[]
    def cost(pred,reference):
        a=pred['softmask'];inter=(a*reference).sum((1,2));union=(a+reference-a*reference).sum((1,2)).clamp_min(1)
        return (1-inter/union+.1*(a-reference).abs().mean((1,2))).mean()
    with torch.no_grad():
        identity=torch.eye(4,device='cuda:0');before=render(vertices,faces,k,w,context,96)
        original_cost=float(cost(before,target))
        for r in rotations:
            pred=render(vertices@r.T,faces,k,w,context,96)
            ranking.append(float(cost(pred,target)))
    target128=F.interpolate(torch.tensor(masks,device='cuda:0',dtype=torch.float32)[:,None],size=(128,128),mode='area')[:,0]
    candidates=[dict(loss=original_cost,rotation_id='identity',step=0,matrix=np.eye(4).tolist())]
    for rid in np.argsort(ranking,kind='stable')[:3]:
        raw=torch.nn.Parameter(torch.zeros(7,device='cuda:0'));optimizer=torch.optim.Adam([raw],lr=.035)
        for step in range(61):
            rotation,scale,translation=similarity(raw,rotations[int(rid)])
            pred=render(scale*(vertices@rotation.T)+translation,faces,k,w,context,128)
            loss=cost(pred,target128)
            if step%10==0:
                matrix=np.eye(4);matrix[:3,:3]=(scale*rotation).detach().cpu().numpy();matrix[:3,3]=translation.detach().cpu().numpy()
                candidates.append(dict(loss=float(loss.detach()),rotation_id=int(rid),step=step,matrix=matrix.tolist()))
            if step<60:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step and step%20==0:emit(index,seed,'fitting',rotation=int(rid),step=step)
        del optimizer,raw
    # Re-score finalists at one common resolution, including the original.
    with torch.no_grad():
        for candidate in candidates:
            m=torch.tensor(candidate['matrix'],device='cuda:0',dtype=torch.float32)
            pred=render(vertices@m[:3,:3].T+m[:3,3],faces,k,w,context,256)
            target256=F.interpolate(torch.tensor(masks,device='cuda:0',dtype=torch.float32)[:,None],size=(256,256),mode='area')[:,0]
            candidate['final_loss']=float(cost(pred,target256))
        chosen=min(candidates,key=lambda row:row['final_loss'])
        matrix=torch.tensor(chosen['matrix'],device='cuda:0',dtype=torch.float32)
        world=vertices@matrix[:3,:3].T+matrix[:3,3];pred=render(world,faces,k,w,context)
        mask=target256>.5;iou=((pred['mask']&mask).sum((1,2))/(pred['mask']|mask).sum((1,2)).clamp_min(1)).cpu().tolist()
        save_mesh(root/'aligned_base.obj',world,faces)
    write(root/'fit.json',dict(time=now(),matrix=chosen['matrix'],chosen=chosen,ious=iou,mean_iou=float(np.mean(iou)),
          coarse_rotation_costs=ranking,seconds=time.monotonic()-start,gt_used=False,proper_rotation_only=True,
          low_registration_quality=float(np.mean(iou))<.4))
    emit(index,seed,'fit_complete',mean_iou=float(np.mean(iou)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup();torch.set_num_threads(2);snapshot('fit')
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0')
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):one(index,seed,context)


if __name__=='__main__':main()
