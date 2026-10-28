"""Fixed native-frame cameras for GT plus the three re-generated arms."""
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from .common import OUT, binding, verify
from ...data_v2.render_scenes import pack_scene, rasterize
from ..hssd_pilot_20261020.scenes import look_at
from ..hssd_pilot_20261020.visuals import shade


@torch.no_grad()
def galleries(rows):
    device, side = torch.device('cuda:0'), 256
    focal = side*.5/np.tan(np.deg2rad(40)*.5)
    k = torch.tensor([[[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]]]*4, device=device, dtype=torch.float32)
    poses = [look_at(2*np.array([np.cos(a)*np.cos(np.pi/6),np.sin(a)*np.cos(np.pi/6),.5]),
        np.zeros(3), np.array([0.,0.,1.])) for a in np.deg2rad([45,135,225,315])]
    w = torch.tensor(np.array(poses),device=device,dtype=torch.float32)
    images = []
    for repeat, records in enumerate(rows):
        lookup = {(v['index'],v['seed'],v['arm']):v for v in records}
        for index in [0,2,5,14]:
            for seed in [42,43,44]:
                arms = ['appearance_control','uniform','balanced']
                reference = lookup[index,seed,arms[0]]
                entries = [('GT',reference['gt'])]+[(a,lookup[index,seed,a]['mesh']) for a in arms]
                canvas = Image.new('RGB',(side*4,side*4+64),'white')
                draw = ImageDraw.Draw(canvas)
                draw.text((8,8),f'Process order {repeat} | index {index} | seed {seed} | shared native-frame cameras',fill='black')
                for col,(label,path) in enumerate(entries):
                    draw.text((col*side+8,34),label,fill='black')
                    mesh = trimesh.load(verify(path),force='mesh',process=False)
                    packed = pack_scene([shade(mesh)],[],device)
                    render = rasterize(packed,k,w,image_size=(side,side),raster_side=side,layers=1,rgb=True,background_rgb=(1.,1.,1.))
                    for view,pixels in enumerate(render['rgb']):
                        canvas.paste(Image.fromarray((pixels.cpu().numpy()*255).round().astype(np.uint8)),(col*side,view*side+64))
                    del packed,render
                path = OUT/'fair/figures'/f'{repeat}_{index:03d}_{seed}.jpg'
                path.parent.mkdir(parents=True,exist_ok=True)
                canvas.save(path,quality=94)
                images.append(binding(path))
    return images
