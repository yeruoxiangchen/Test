"""Fixed cameras shared by GT and all five methods, without alignment."""
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from ...data_v2.render_scenes import pack_scene, rasterize
from ..hssd_pilot_20261020.scenes import look_at
from ..hssd_pilot_20261020.visuals import shade
from .common import OUT, INDICES, SEEDS, binding


@torch.no_grad()
def galleries(rows):
    device=torch.device('cuda:0');side=256
    focal=side*.5/np.tan(np.deg2rad(40)*.5)
    k=torch.tensor([[[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]]]*4,device=device,dtype=torch.float32)
    poses=[]
    for angle in np.deg2rad([45,135,225,315]):
        eye=2*np.array([np.cos(angle)*np.cos(np.pi/6),np.sin(angle)*np.cos(np.pi/6),.5])
        poses.append(look_at(eye,np.zeros(3),np.array([0.,0.,1.])))
    w=torch.tensor(np.array(poses),device=device,dtype=torch.float32)
    arms=list(rows);by_key={a:{(r['index'],r['seed']):r for r in rr} for a,rr in rows.items()}
    result=[]
    for index in INDICES:
        for seed in SEEDS:
            dest=OUT/'figures'/f'{index:03d}_{seed}.jpg'
            canvas=Image.new('RGB',(side*(1+len(arms)),side*4+64),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f'Occluded | global index {index} | seed {seed} | native coordinates, shared cameras',fill='black')
            entries=[('GT',by_key['base'][index,seed]['gt']['path'])]+[
                (a,by_key[a][index,seed].get('mesh',{}).get('path')) for a in arms]
            for col,(label,path) in enumerate(entries):
                draw.text((col*side+8,34),label,fill='black')
                if not path:
                    draw.text((col*side+8,90),'FAILED / no mesh',fill='red');continue
                mesh=trimesh.load(path,force='mesh',process=False)
                packed=pack_scene([shade(mesh)],[],device)
                rendered=rasterize(packed,k,w,image_size=(side,side),raster_side=side,layers=1,rgb=True,background_rgb=(1.,1.,1.))
                for view,pixels in enumerate(rendered['rgb']):
                    canvas.paste(Image.fromarray((pixels.numpy()*255).round().astype(np.uint8)),(col*side,view*side+64))
                del packed,rendered
            dest.parent.mkdir(exist_ok=True);canvas.save(dest,quality=94);result.append(binding(dest))
    return result
