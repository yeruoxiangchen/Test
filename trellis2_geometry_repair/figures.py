import argparse
import textwrap

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from .config import OUT,PARAMS,folder,read,write,setup,now
from .geometry import load_mesh,render,observations


def shade(pred,view):
    normal=pred['normal'][view].cpu().numpy()
    light=np.array([.25,-.45,.85]);light=light/np.linalg.norm(light)
    intensity=.35+.65*np.abs((normal*light).sum(-1))
    rgb=intensity[...,None]*np.array([135,160,180])[None,None]
    return np.where(pred['mask'][view].cpu().numpy()[...,None],rgb,255).clip(0,255).astype(np.uint8)


def one(index,seed,context):
    root=folder(index,seed);result=read(root/'result.json')
    camera=read(OUT/'data'/f'{index:03d}'/'evaluation_cameras.json')
    k=torch.tensor(camera['intrinsics'],device='cuda:0');w=torch.tensor(camera['extrinsics'],device='cuda:0')
    gt=torch.load(OUT/'data'/f'{index:03d}'/'gt_evaluation_render.pt',weights_only=False)
    views=[0,2,5];rows=[('GT scan\nEvaluation only',gt)]
    for label,path in [('TRELLIS.2\nRaw native',root/'native_original.obj'),
                       ('TRELLIS.2\nInput aligned',root/'aligned_base.obj'),
                       ('Local SLat repair\nDelivered',root/'accepted.obj'),
                       ('Local SLat repair\nCandidate',root/'candidate.obj')]:
        v,f=load_mesh(path);rows.append((label,render(v,f,k,w,context)))
    crops=[]
    for view in views:
        mask=np.logical_or.reduce([pred['mask'][view].cpu().numpy() for _,pred in rows]);yy,xx=np.nonzero(mask)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4)
            x=max(0,min(256-width,int((xx.max()+xx.min()-width)/2)));y=max(0,min(256-width,int((yy.max()+yy.min()-width)/2)))
            crops.append((x,y,x+width,y+width))
        else:crops.append((0,0,256,256))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    canvas=Image.new('RGB',(1140,160+len(rows)*290),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,20),f'{result["uid"]}, seed {seed} | native TRELLIS.2 Shape-512',font=font,fill='black')
    draw.text((20,50),'Same cameras and shared crop | GT used only for evaluation',font=font,fill='black')
    for ri,(label,pred) in enumerate(rows):
        y=100+ri*290;draw.multiline_text((12,y+100),label,font=font,fill='black',spacing=5)
        for col,(view,box) in enumerate(zip(views,crops)):
            tile=Image.fromarray(shade(pred,view)).crop(box).resize((280,280),Image.Resampling.BILINEAR)
            canvas.paste(tile,(270+col*285,y))
    dest=OUT/'figures'/f'{index:03d}_{seed}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest,quality=94)
    _,_,_,images,_=observations(index)
    inputs=Image.new('RGB',(4*256,280),'white');draw=ImageDraw.Draw(inputs)
    for view,image in enumerate(images):
        inputs.paste(image.resize((256,256)),(view*256,24));draw.text((view*256+3,2),f'Input {view}',font=font,fill='black')
    inputs.save(OUT/'figures'/f'{index:03d}_inputs.jpg',quality=94)
    return str(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');a=p.parse_args();setup()
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0');path=OUT/'figures/manifest.json';manifest=read(path) if path.exists() else dict(cases={})
    with torch.no_grad():
        for index in map(int,a.indices.split(',')):
            for seed in PARAMS['seeds']:
                key=f'{index:03d}_{seed}'
                if key in manifest['cases']:continue
                manifest['cases'][key]=one(index,seed,context);write(path,manifest)
    manifest.update(time=now(),complete=len(manifest['cases'])==8);write(path,manifest)


if __name__=='__main__':main()
