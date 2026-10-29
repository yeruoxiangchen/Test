import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from .config import BASE,DIAGNOSTIC,OUT,ARMS,case,source,read,write,digest,setup,lock_case,now
from .fields import lookup
from ...geometry import load_mesh,render,error,transform
from ...metrics import surface_scores
from ...figures import shade
from ..repair_diagnostic.data import evaluation_points


def figure(index,seed,records,scope):
    views=[0,2,5];crops=[]
    for view in views:
        mask=np.logical_or.reduce([p['mask'][view].cpu().numpy() for _,p in records]);yy,xx=np.nonzero(mask)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.12)+4)
            x=max(0,min(256-width,int((xx.max()+xx.min()-width)/2)));y=max(0,min(256-width,int((yy.max()+yy.min()-width)/2)))
            crops.append((x,y,x+width,y+width))
        else:crops.append((0,0,256,256))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    canvas=Image.new('RGB',(1140,120+len(records)*285),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,12),f'{index:03d}, seed {seed} | {scope} native Flow resampling | common cameras/crops',font=font,fill='black')
    draw.text((12,38),'Delivered rows include fallbacks. Candidate native/FDG composite are explicitly separate.',font=font,fill='black')
    for ri,(label,pred) in enumerate(records):
        y=90+ri*285;draw.multiline_text((10,y+95),label,font=font,fill='black',spacing=5)
        for col,(view,box) in enumerate(zip(views,crops)):
            tile=Image.fromarray(shade(pred,view)).crop(box).resize((275,275),Image.Resampling.BILINEAR)
            canvas.paste(tile,(270+col*285,y))
    path=OUT/'figures'/f'{index:03d}_{seed}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=94)
    with (path.parent/'manifest.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        manifest=path.parent/'manifest.json';value=read(manifest) if manifest.exists() else dict(cases={})
        value['cases'][f'{index:03d}_{seed}']=str(path);value.update(time=now(),complete=len(value['cases'])==8);write(manifest,value)
    return str(path)


def one(index,seed,context):
    root=case(index,seed)
    if (root/'result.json').exists():return
    selected=read(root/'input_complete.json')
    if selected['gt_used']:raise RuntimeError('input decision used GT')
    write(root/'input_sealed.json',dict(time=now(),decision_sha256=digest(root/'input_complete.json'),gt_used=False))
    target_points=evaluation_points(index)
    camera=read(BASE/'data'/f'{index:03d}'/'evaluation_cameras.json')
    k=torch.tensor(camera['intrinsics'],device='cuda:0');w=torch.tensor(camera['extrinsics'],device='cuda:0')
    gt={key:value.cuda() for key,value in torch.load(BASE/'data'/f'{index:03d}'/'gt_evaluation_render.pt',weights_only=False).items()}
    old=torch.load(DIAGNOSTIC/'objects'/f'{index:03d}'/str(seed)/'baseline_geometry.pt',weights_only=False)
    ov,of=old['vertices'].cuda(),old['faces'].cuda();old_pred=render(ov,of,k,w,context)
    bv,bf=load_mesh(root/'registered_base.obj');base_pred=render(bv,bf,k,w,context)
    plan=torch.load(root/'plan.pt',weights_only=False);matrix=plan['matrix'].cuda()
    envelope=torch.tensor(plan['roi']['envelope'],device='cuda:0')
    native=transform(base_pred['xyz'].reshape(-1,3),torch.linalg.inv(matrix))
    region=lookup(envelope,torch.floor((native+.5)*32).long()).reshape(base_pred['mask'].shape).bool()&base_pred['mask']
    def domain(pred):
        valid=pred['mask']|gt['mask'];bad=valid&(error(pred,gt)>=1/256)
        return dict(all=valid,bad=bad,good=valid&~bad)
    old_masks=domain(old_pred);masks=domain(base_pred);masks['selected_bad']=region&masks['bad']
    def score(v,f):
        pred=render(v,f,k,w,context);err=error(pred,gt)
        values=lambda m:{name:float(err[mask].mean()) if bool(mask.any()) else None for name,mask in m.items()}
        return dict(surface=surface_scores(v.cpu().numpy(),f.cpu().numpy(),target_points),
                    novel_depth_registered_domain=values(masks),novel_depth_previous_domain=values(old_masks)),pred
    old_scores,_=score(ov,of);base_scores,_=score(bv,bf)
    previous=read(source(index,seed)/'result.json')['original']['surface']['chamfer_l1']
    if abs(old_scores['surface']['chamfer_l1']-previous)>1.e-7:raise RuntimeError('previous baseline CD replay mismatch')
    cache={(root/'registered_base.obj').resolve():(base_scores,base_pred)}
    endpoints={}
    for arm in ARMS:
        endpoints[arm]={}
        for ep in ('accepted','candidate'):
            p=(root/arm/f'{ep}.obj').resolve()
            if p not in cache:
                v,f=load_mesh(p);cache[p]=score(v,f)
            endpoints[arm][ep]=cache[p][0]
    adaptive=selected['selections']['adaptive']['candidate']
    paired=[]
    for key in ('native_mesh','composite_mesh'):
        p=Path(adaptive.get(key,adaptive['mesh'])).resolve()
        if p not in cache:
            v,f=load_mesh(p);cache[p]=score(v,f)
        paired.append(cache[p][1])
    rows=[('GT scan\nEvaluation only',gt),('Previous input-aligned\nStock baseline',old_pred),
          ('New registration only\nStock baseline',base_pred),
          ('Shape resampling\nDELIVERED',cache[(root/'shape/accepted.obj').resolve()][1]),
          ('Adaptive SS / Shape\nDELIVERED',cache[(root/'adaptive/accepted.obj').resolve()][1]),
          ('Adaptive candidate\nDIRECT NATIVE decode',paired[0]),
          ('Same candidate\nFDG COMPOSITE',paired[1])]
    result=dict(time=now(),index=index,seed=seed,uid=selected['uid'],previous_baseline=old_scores,registered_baseline=base_scores,
          endpoints=endpoints,selections=selected['selections'],plan=read(root/'plan.json'),
          gt_used_for_input_decisions=False,previous_cd_replay_abs_error=abs(old_scores['surface']['chamfer_l1']-previous),
          field_composite_is_not_direct_latent_decode=True,figure=figure(index,seed,rows,selected['scope']),
          selection_seconds=selected.get('seconds'),selection_peak_gib=selected.get('peak_gib'))
    write(root/'result.json',result);print(dict(index=index,seed=seed,stage='evaluated_complete'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed),torch.no_grad():one(a.index,a.seed,dr.RasterizeCudaContext(device='cuda:0'))


if __name__=='__main__':main()
