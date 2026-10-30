"""Shared-camera native mesh comparisons using a small independent rasterizer."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from . import common as c


def cameras(device,side=288):
    matrices=[]
    for az,el in [(30,20),(150,20),(270,35)]:
        az,el=np.deg2rad([az,el])
        eye=1.9*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        forward=-eye/np.linalg.norm(eye)
        right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        r=np.stack([right,down,forward])
        w=np.eye(4,dtype=np.float32);w[:3,:3]=r;w[:3,3]=-r@eye
        matrices.append(w)
    return torch.tensor(np.array(matrices),device=device)


@torch.no_grad()
def mesh_tiles(path,context,side=288):
    import trimesh
    import nvdiffrast.torch as dr
    mesh=path if isinstance(path,trimesh.Trimesh) else trimesh.load(path,force='mesh',process=False,skip_materials=True)
    v=torch.tensor(np.asarray(mesh.vertices),device='cuda',dtype=torch.float32)
    f=torch.tensor(np.asarray(mesh.faces),device='cuda',dtype=torch.int32)
    fn=torch.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]],dim=-1)
    normal=torch.zeros_like(v)
    for j in range(3):
        normal.index_add_(0,f[:,j].long(),fn)
    normal=torch.nn.functional.normalize(normal,dim=-1,eps=1e-10)
    w=cameras(v.device,side)
    vh=torch.cat([v,torch.ones_like(v[:,:1])],-1)
    cam=torch.einsum('vij,nj->vni',w,vh)[...,:3]
    near,far=.05,5.
    clip=torch.stack([2*320/side*cam[...,0],-2*320/side*cam[...,1],
        (far+near)/(far-near)*cam[...,2]-2*far*near/(far-near),cam[...,2]],-1)
    raster,_=dr.rasterize(context,clip.contiguous(),f,resolution=(side,side))
    attr,_=dr.interpolate(normal[None].expand(3,-1,-1).contiguous(),raster,f)
    attr=torch.nn.functional.normalize(attr,dim=-1,eps=1e-8)
    light=torch.tensor([.25,-.45,.85],device='cuda');light/=light.norm()
    intensity=.35+.65*(attr*light).sum(-1).abs()
    color=torch.tensor([135.,160.,180.],device='cuda')
    rgb=intensity[...,None]*color
    rgb=torch.where(raster[...,3:4]>0,rgb,torch.full_like(rgb,255))
    # RasterizeCuda uses bottom-up image rows; display convention is top-down.
    return [Image.fromarray(x) for x in rgb.flip(1).clamp(0,255).byte().cpu().numpy()]


def one(folder,variant,context):
    folder=Path(folder)
    row=c.read(folder/f'shape_metrics_{variant}.json')
    entries=[('Raw GT','gt.obj'),('Native codec GT',f'native_codec_gt_{variant}.obj'),
             ('Zero features / GT support',f'damaged_{variant}.obj'),('Stock local flow',f'stock_local_{variant}.obj'),
             ('Repair / teacher support',f'candidate_teacher_support_{variant}.obj')]
    if (folder/f'damaged_input_{variant}.obj').exists():
        entries.insert(2,('Actual damaged input',f'damaged_input_{variant}.obj'))
    if (folder/f'candidate_predicted_ss_{variant}.obj').exists():
        entries.append(('Predicted SS rollout',f'candidate_predicted_ss_{variant}.obj'))
    if (folder/f'stock_predicted_ss_{variant}.obj').exists():
        entries.insert(len(entries)-1,('Stock SS + stock Shape',f'stock_predicted_ss_{variant}.obj'))
    side=288
    canvas=Image.new('RGB',(1130,90+len(entries)*300),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    draw.text((15,10),row['uid']+' / '+row['kind'],fill='black',font=font)
    draw.text((15,38),'Same native coordinates and fixed cameras; known synthetic ROI; no per-method GT fit',fill='black',font=small)
    for i,(label,path) in enumerate(entries):
        y=80+i*300
        draw.text((10,y+80),label,fill='black',font=small)
        cached=[folder/f'tile_{Path(path).stem}_{j}.png' for j in range(3)]
        tiles=([Image.open(p).convert('RGB') for p in cached] if all(p.exists() for p in cached)
               else mesh_tiles(folder/path,context,side))
        for j,tile in enumerate(tiles):
            canvas.paste(tile,(255+j*290,y))
    dest=folder/f'comparison_{variant}.jpg'
    canvas.save(dest,quality=93)
    return dest


def gallery(root,paths):
    root=Path(root).resolve()
    if not root.is_relative_to(c.ROOT):
        raise ValueError('Figure output outside project')
    import html
    body=['<!doctype html><meta charset="utf-8"><title>Native T2 repair audit</title>',
          '<h1>Native T2 repair: shared-camera comparisons</h1>',
          '<p>Controlled synthetic interventions; known ROI; teacher-support and predicted-SS results are distinct.</p>']
    for p in paths:
        rel=Path(p).relative_to(root)
        body.append(f'<h2>{html.escape(str(rel.parent))}</h2><img style="max-width:1130px;width:100%" loading="lazy" src="{html.escape(str(rel))}">')
    (root/'index.html').write_text('\n'.join(body))
    c.write(root/'figures.json',{'time':c.now(),'images':[str(p) for p in paths],'complete':True})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',required=True)
    args=p.parse_args()
    root=Path(args.evaluation).resolve()
    if not root.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Evaluation directory outside project')
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0')
    paths=[]
    for file in sorted((root/'objects').glob('*/shape_metrics_*.json')):
        paths.append(one(file.parent,int(file.stem.split('_')[-1]),context))
    gallery(root,paths)


if __name__=='__main__':
    main()
