import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from .config import OUT,BASE,ARMS,DIAGNOSTIC,case,previous,read,write,digest,now,setup,lock_case
from ...geometry import load_mesh,render,error
from ...metrics import surface_scores
from ...figures import shade
from ..repair_diagnostic.data import evaluation_points

def fixed_depth_metrics(pred,gt,base):
    domain=base['mask']&gt['mask'];hit=domain&pred['mask']
    residual=(pred['depth']-gt['depth']).abs()
    fixed=torch.where(pred['mask'],residual,torch.full_like(residual,.1))
    return dict(fixed_queries=int(domain.sum()),hits=int(hit.sum()),
        common_hit_mae=float(residual[hit].mean()) if bool(hit.any()) else None,
        fixed_unclipped_depth_with_missing_penalty=float(fixed[domain].mean()) if bool(domain.any()) else None,
        lost_fraction=float((domain&~pred['mask']).sum()/domain.sum().clamp_min(1)),missing_penalty=.1)

def figure(index,seed,rows):
    views=[0,2,5];boxes=[]
    for view in views:
        mask=np.logical_or.reduce([p['mask'][view].cpu().numpy() for _,p in rows]);yy,xx=np.nonzero(mask)
        width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.12)+4) if len(yy) else 256
        x=max(0,min(256-width,int((xx.max()+xx.min()-width)/2))) if len(yy) else 0
        y=max(0,min(256-width,int((yy.max()+yy.min()-width)/2))) if len(yy) else 0;boxes.append((x,y,x+width,y+width))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    canvas=Image.new('RGB',(1140,100+285*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,12),f'{index:03d}, seed {seed} | V3 native geometry-guided Flow | identical pose/cameras/crops',font=font,fill='black')
    draw.text((12,38),'GT/paired arms are DIAGNOSTICS. Delivered includes fallbacks. Composite is separately labeled.',font=font,fill='black')
    for ri,(label,pred) in enumerate(rows):
        y=80+285*ri;draw.multiline_text((10,y+85),label,font=font,fill='black',spacing=5)
        for col,(view,box) in enumerate(zip(views,boxes)):
            tile=Image.fromarray(shade(pred,view)).crop(box).resize((275,275),Image.Resampling.BILINEAR);canvas.paste(tile,(270+285*col,y))
    path=OUT/'figures'/f'{index:03d}_{seed}.jpg';path.parent.mkdir(exist_ok=True);canvas.save(path,quality=94);return str(path)

def one(index,seed,context):
    root=case(index,seed)
    if (root/'result.json').exists():return
    method=read(root/'method_complete.json');diag=read(root/'diagnostic_complete.json')
    if method['GT_used'] or digest(root/'method_complete.json')!=read(root/'method_sealed.json')['decision_sha256']:raise RuntimeError('method is not sealed input-only')
    if diag['domains']['paired_input']['query_ids_sha256']!=diag['domains']['oracle']['query_ids_sha256']:raise RuntimeError('paired diagnostic query IDs differ')
    cameras=read(BASE/'data'/f'{index:03d}'/'evaluation_cameras.json')
    k=torch.tensor(cameras['intrinsics'],device='cuda:0');w=torch.tensor(cameras['extrinsics'],device='cuda:0')
    gt={k:v.cuda() for k,v in torch.load(BASE/'data'/f'{index:03d}'/'gt_evaluation_render.pt',weights_only=False).items()}
    target=evaluation_points(index);bv,bf=load_mesh(previous(index,seed)/'registered_base.obj');base=render(bv,bf,k,w,context)
    valid=base['mask']|gt['mask'];bad=valid&(error(base,gt)>=1/256);good=valid&~bad
    def score(v,f):
        pred=render(v,f,k,w,context);err=error(pred,gt)
        metrics=dict(surface=surface_scores(v.cpu().numpy(),f.cpu().numpy(),target),
            novel_clipped_error={name:float(err[mask].mean()) if bool(mask.any()) else None for name,mask in dict(all=valid,bad=bad,good=good).items()},
            novel_unclipped=fixed_depth_metrics(pred,gt,base))
        return metrics,pred
    base_score,_=score(bv,bf)
    expected=read(previous(index,seed)/'result.json')['registered_baseline']['surface']['chamfer_l1']
    if abs(base_score['surface']['chamfer_l1']-expected)>1.e-7:raise RuntimeError('frozen V2 base metric replay differs')
    cache={(previous(index,seed)/'registered_base.obj').resolve():(base_score,base)}
    def get(path):
        path=Path(path).resolve()
        if path not in cache:
            v,f=load_mesh(path);cache[path]=score(v,f)
        return cache[path]
    v2=read(previous(index,seed)/'input_complete.json')['selections']['adaptive']['accepted'];v2_score,v2_pred=get(v2['mesh'])
    endpoints={};selections=method['selections']|diag['selections']
    for arm in ARMS:
        endpoints[arm]={ep:get(selections[arm][ep]['mesh'])[0] for ep in ('accepted','candidate')}
    selected=selections['input']['candidate'];composite=get(selected.get('composite_mesh',selected['mesh']))[1]
    rows=[('GT scan\nEvaluation only',gt),('Stock base\nFrozen input alignment',base),('V2 adaptive\nDELIVERED',v2_pred)]
    for arm,label in [('prior','Prior, 2 proposals\nDELIVERED'),('budget_prior','Prior, measured budget\nDELIVERED'),('input','V3 input-guided\nDELIVERED')]:
        rows.append((label,get(selections[arm]['accepted']['mesh'])[1]))
    rows += [('V3 input candidate\nDIRECT NATIVE',get(selections['input']['candidate']['mesh'])[1]),
             ('Same input candidate\nFDG COMPOSITE, DIAG',composite),
             ('Paired input candidate\nGT-domain DIAGNOSTIC',get(selections['paired_input']['candidate']['mesh'])[1]),
             ('GT oracle candidate\nDIAGNOSTIC ONLY',get(selections['oracle']['candidate']['mesh'])[1])]
    result=dict(time=now(),index=index,seed=seed,uid=method['uid'],baseline=base_score,v2=v2_score,endpoints=endpoints,selections=selections,
        budget=method['budget'],domains=method['domains']|diag['domains'],GT_used_for_method=False,diagnostic_arms=list(DIAGNOSTIC),
        base_cd_replay_error=abs(base_score['surface']['chamfer_l1']-expected),figure=figure(index,seed,rows))
    write(root/'result.json',result);print(dict(index=index,seed=seed,evaluated_complete=True),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed),torch.no_grad():one(a.index,a.seed,dr.RasterizeCudaContext(device='cuda:0'))

if __name__=='__main__':main()
