import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from .config import BASE,OUT,ARMS,case,original,read,write,now,setup,case_lock
from .data import load,oracle,seal_inputs,evaluation_points
from ...geometry import load_mesh,render,error
from ...metrics import surface_points,surface_scores
from ...figures import shade


def evidence_quality(data,gt_target):
    queries=data['target']['depth_valid']&(data['region']>.5)
    common=queries & gt_target['mask'] & data['baseline']['mask']
    gt=gt_target['depth'];teacher=data['target']['depth'];base=data['baseline']['depth']
    a=(teacher-gt)[common];b=(base-gt)[common]
    if not len(a):return dict(common_rays=0,requested_rays=int(queries.sum()))
    return dict(common_rays=len(a),requested_rays=int(queries.sum()),
        common_support_fraction=float(common.sum()/queries.sum().clamp_min(1)),
        target_raw_mae=float(a.abs().mean()),base_raw_mae=float(b.abs().mean()),
        target_signed_bias=float(a.mean()),target_better_ray_fraction=float((a.abs()<b.abs()).float().mean()),
        target_depth_ratio_median=float((teacher[common]/gt[common]).median()),
        target_source='frozen input-derived mixed DA3/NCC; GT audit only',
        gt_used_to_calibrate_input=False)


def figure(index,seed,records):
    views=[0,2,5];crops=[]
    for view in views:
        mask=np.logical_or.reduce([pred['mask'][view].cpu().numpy() for _,pred in records]);yy,xx=np.nonzero(mask)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4)
            x=max(0,min(256-width,int((xx.max()+xx.min()-width)/2)));y=max(0,min(256-width,int((yy.max()+yy.min()-width)/2)))
            crops.append((x,y,x+width,y+width))
        else:crops.append((0,0,256,256))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    canvas=Image.new('RGB',(1140,130+len(records)*290),'white');draw=ImageDraw.Draw(canvas)
    draw.text((15,15),f'{index:03d}, seed {seed}: same cameras, shared input-only alignment and crop',font=font,fill='black')
    draw.text((15,42),'Editors shown at final step (unfiltered proposals). ORACLE rows are diagnostics, not method results.',font=font,fill='black')
    for ri,(label,pred) in enumerate(records):
        y=95+ri*290;draw.multiline_text((12,y+100),label,font=font,fill='black',spacing=5)
        for col,(view,box) in enumerate(zip(views,crops)):
            tile=Image.fromarray(shade(pred,view)).crop(box).resize((280,280),Image.Resampling.BILINEAR)
            canvas.paste(tile,(270+col*285,y))
    path=OUT/'figures'/f'{index:03d}_{seed}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=94)
    manifest=OUT/'figures/manifest.json'
    with (manifest.parent/'manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=read(manifest) if manifest.exists() else dict(cases={})
        value['cases'][f'{index:03d}_{seed}']=str(path);value.update(time=now(),complete=len(value['cases'])==8);write(manifest,value)
    return str(path)


def one(index,seed,context):
    root=case(index,seed)
    if (root/'result.json').exists():return
    if not all((root/arm/'complete.json').exists() for arm in ARMS):raise RuntimeError('four sealed arms required')
    seal_inputs(index,seed);data=load(index,seed,context)
    gt_input,gv,gf=oracle(index,seed,data,context);quality=evidence_quality(data,gt_input)
    target_points=evaluation_points(index)
    camera=read(BASE/'data'/f'{index:03d}'/'evaluation_cameras.json')
    k=torch.tensor(camera['intrinsics'],device='cuda:0');w=torch.tensor(camera['extrinsics'],device='cuda:0')
    gt={key:value.cuda() for key,value in torch.load(BASE/'data'/f'{index:03d}'/'gt_evaluation_render.pt',weights_only=False).items()}
    baseline=render(data['vertices'],data['faces'],k,w,context)
    with np.load(original(index,seed)/'evidence_sealed.npz') as z:selected=torch.tensor(z['selected_faces'],device='cuda:0')
    local=selected[baseline['face_id'].clamp_min(0)]&baseline['mask']
    domain=baseline['mask']|gt['mask'];initial_error=error(baseline,gt)
    bad=domain&(initial_error>=1/256);good=domain&~bad
    masks=dict(all=domain,bad=bad,good=good,selected_bad=local&bad)
    def score(v,f):
        pred=render(v,f,k,w,context);errors=error(pred,gt)
        metrics=dict(surface=surface_scores(v.cpu().numpy(),f.cpu().numpy(),target_points),
                     novel_depth={name:float(errors[mask].mean()) if bool(mask.any()) else None for name,mask in masks.items()})
        return metrics,pred
    base_scores,base_pred=score(data['vertices'],data['faces'])
    old=read(original(index,seed)/'result.json')['original']['surface']['chamfer_l1']
    if abs(base_scores['surface']['chamfer_l1']-old)>1.e-7:
        raise RuntimeError(f'baseline surface metric drift: {base_scores["surface"]["chamfer_l1"]} vs {old}')
    cache={(original(index,seed)/'aligned_base.obj').resolve():(base_scores,base_pred)}
    records=[('GT scan\nEvaluation / oracle only',gt),('TRELLIS.2\nShared input-aligned base',base_pred)]
    values={}
    for arm in ARMS:
        selection=read(root/arm/'complete.json');endpoints={}
        for ep in ('accepted','candidate','proposal'):
            path=(root/arm/f'{ep}.obj').resolve()
            if path not in cache:
                v,f=load_mesh(path);cache[path]=score(v,f)
            endpoints[ep]=cache[path][0]
        final=(root/arm/'proposal.obj').resolve()
        label=arm.replace('_',' ').upper() if arm.endswith('oracle') else arm.replace('_',' ')
        label+=f'\nFinal proposal; accept step {selection["decisions"]["accepted_step"]}'
        records.append((label,cache[final][1]))
        values[arm]=dict(oracle=arm.endswith('oracle'),endpoints=endpoints,
                        decisions=selection['decisions'],seconds=selection['seconds'],peak_gib=selection['peak_gib'],
                        representation=selection['representation'],snapshots=selection['snapshots'])
    receipt=dict(time=now(),index=index,seed=seed,uid=data['uid'],baseline=base_scores,arms=values,
                 evidence_quality=quality,gt_used_for_input_training_or_selection=False,
                 baseline_cd_replay_abs_error=abs(base_scores['surface']['chamfer_l1']-old),
                 shared_global_registration=True,development_diagnostic=True)
    receipt['figure']=figure(index,seed,records)
    write(root/'result.json',receipt);print(dict(index=index,seed=seed,stage='evaluated_complete'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65)
    import nvdiffrast.torch as dr
    with case_lock(a.index,a.seed),torch.no_grad():
        one(a.index,a.seed,dr.RasterizeCudaContext(device='cuda:0'))


if __name__=='__main__':main()
