"""Reuse the audited native-frame renderer, with identical cameras for all arms."""
from PIL import Image, ImageDraw
from .common import *


def make_figures(rows):
    from ..round2_v2.render_mesh_audit import render_file
    frozen=plan()
    root=OUT/'visuals'
    records={}
    for obj in frozen['plan']['objects']:
        uid=obj['uid']
        event('render_mesh_gallery',uid=uid)
        records[uid,'gt',None]=render_file(verify(obj['mesh']),uid+'_gt',output=root,
            expected_sha256=obj['mesh']['sha256'])
        for row in rows:
            if row['object_uid']!=uid:continue
            records[uid,row['arm'],row['seed']]=render_file(verify(row['mesh']),
                f"{uid}_{row['arm']}_{row['seed']}",output=root,expected_sha256=row['mesh']['sha256'])
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    links=[]
    for obj in frozen['plan']['objects']:
        uid=obj['uid']
        for seed in SEEDS:
            side,header=320,28
            sheet=Image.new('RGB',(side*3,(side+header)*3+header),'white')
            draw=ImageDraw.Draw(sheet)
            draw.text((8,6),f'{uid} / seed {seed} / native frame, no fitted transform',fill='black')
            for r,arm in enumerate(['gt',*ARMS]):
                record=records[uid,arm,None if arm=='gt' else seed]
                for c,view in enumerate(record['views']):
                    y=header+r*(header+side)
                    draw.text((c*side+8,y+6),f'{arm} / view {c}',fill='black')
                    with Image.open(view['png']) as image:
                        sheet.paste(image.resize((side,side)),(c*side,y+header))
            dest=figures/f'{uid}_{seed}.jpg'
            sheet.save(dest,quality=95)
            links.append(binding(dest))
    clipping=[{'uid':uid,'arm':arm,'seed':seed,'view':v['view']}
        for (uid,arm,seed),record in records.items() for v in record['views']
        if v['foreground_touches_image_border'] or v['vertices_outside_xy'] or v['vertices_outside_depth']]
    write(OUT/'visual_audit.json',{'time':now(),'figures':links,'clipping':clipping,
        'camera_policy':'three fixed native orthographic cameras; same material and scale; no fitting',
        'review_required':True})
    return links
