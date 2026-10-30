"""CPU-only source-frame audit: geometry projection versus material visibility."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import maximum_filter
from . import common as c
from .evaluate import canonical_gt
from .prepare import voxel_projection_radius


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--uids',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output).resolve()
    if not out.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Audit output outside project')
    out.mkdir(parents=True,exist_ok=True)
    rows={r['object_uid']:r for r in c.plan()['plan']['samples']}
    results=[]
    for uid in args.uids:
        row=rows[uid]
        np.random.seed(20261030)
        points=canonical_gt(row).sample(50000)
        points=np.c_[points,np.ones(len(points))]
        rot=np.eye(4);rot[:3,:3]=row['canonical']['latent_to_render_rotation']
        modes={'given':np.eye(4),'remove_source_R':np.linalg.inv(rot),'double_source_R':rot}
        canvas=Image.new('RGB',(1536,3*550),'white')
        draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        records=[]
        for vi,view_index in enumerate([0,3,6]):
            view=row['views'][view_index]
            rgb=Image.open(view['image']).convert('RGB')
            mask=np.asarray(Image.open(view['mask']).convert('L'))/255.
            k=np.array(view['intrinsic_normalized']);k[:2]*=512
            for mi,(name,a) in enumerate(modes.items()):
                camera=(points@(np.asarray(view['extrinsic_w2c'])@a).T)[:,:3]
                uv=camera[:,:2]/camera[:,2:3]*np.array([k[0,0],k[1,1]])+k[:2,2]
                valid=(camera[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<=511)&(uv[:,1]>=0)&(uv[:,1]<=511)
                xy=np.rint(uv[valid]).astype(int)
                radius=voxel_projection_radius(camera[valid],k)
                dilated=maximum_filter(mask,size=2*radius+1,mode='constant')
                inside=mask[xy[:,1],xy[:,0]]>.8
                score=float(inside.mean())
                tolerant=float(dilated[xy[:,1],xy[:,0]].mean())
                rec={'uid':uid,'view':view_index,'mode':name,'true_surface_mask_support':score,
                    'tolerant_support':tolerant,'radius':radius,'frustum_fraction':float(valid.mean())}
                records.append(rec)
                overlay=Image.new('RGBA',rgb.size,(0,0,0,0));pen=ImageDraw.Draw(overlay)
                for q,good in zip(xy[::8],inside[::8]):
                    pen.point(tuple(q),fill=(0,255,0,160) if good else (255,0,0,160))
                tile=Image.alpha_composite(rgb.convert('RGBA'),overlay).convert('RGB')
                canvas.paste(tile,(mi*512,vi*550+35))
                draw.text((mi*512+5,vi*550+6),f'{name}, v{view_index}: raw={score:.3f} tolerant={tolerant:.3f}',font=font,fill='black')
        canvas.save(out/f'{uid}.jpg',quality=94)
        results.extend(records)
    c.write(out/'projection_audit.json',{'time':c.now(),'records':results,
        'scope':'GT source geometry and alternate fixed source rotations used only for data diagnosis; no method selection or GT alignment'})


if __name__=='__main__':
    main()
