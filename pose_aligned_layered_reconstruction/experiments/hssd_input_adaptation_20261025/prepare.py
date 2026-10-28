"""Freeze observation-only view selection and shared-camera ROI definitions."""
import numpy as np
from PIL import Image,ImageDraw
from .common import *
from .geometry import projected_cube_box,crop_affine,resize_affine,projection_check,crop_image
from ..hssd_shared_scene_20261025.geometry import canonical_extrinsics

def prepare():
    old=read(SHARED/'plan.json');original=old['plan']
    if old['identity']!=digest(original):raise ValueError('Original shared plan changed')
    verify(original['blend'])
    evaluation=read(SHARED/'evaluation_identity.json')
    for value in evaluation['sources']:verify(value)
    verify(evaluation['manifest']);verify(evaluation['checkpoint'])
    rows_by_uid={r['object_uid']:r for r in read(SHARED/'manifest.json')['samples']}
    objects=[];checks=[]
    for obj in original['objects']:
        row=dict(obj);uid=row['uid'];source=rows_by_uid[uid]
        row['source_sample']={'path':source['sample'],'sha256':source['sha256']};verify(row['source_sample'])
        w=canonical_extrinsics(np.array(original['world_w2c'])[FRAMES],row['canonical_from_world'])
        counts=[];views=[]
        for j,f in enumerate(FRAMES):
            k=np.array(original['intrinsics'][f])
            mask_path=SHARED/'masks'/uid/f'{f:03d}.png';rgb_path=SHARED/'rgb'/f'{f:03d}.png'
            mask=np.asarray(Image.open(mask_path).convert('L'))>127
            count=int(mask.sum());counts.append(count)
            box=projected_cube_box(k,w[j]);a=crop_affine(box)
            high_box=[x*HIGH_SCALE for x in box]
            high_a=crop_affine(high_box);high_k=resize_affine(HIGH_SCALE)@k
            err=float(np.abs(high_a@high_k-a@k).max())
            if err>1.e-9:raise ValueError('Low/high ROI intrinsics differ')
            ys,xs=np.nonzero(mask)
            outside=int(((xs<box[0])|(xs>=box[2])|(ys<box[1])|(ys>=box[3])).sum())
            if outside:raise ValueError('Projected cube crop excludes visible target pixels')
            views.append({'frame':f,'box_518':box,'box_2072':high_box,'affine_518':a.tolist(),
                'intrinsics':(a@k).tolist(),'extrinsics':w[j].tolist(),
                'source_rgb':binding(rgb_path),'source_mask':binding(mask_path),
                'visible_pixels_518':count})
            checks.append({'uid':uid,'frame':f,'crop_projection_error_pixels':projection_check(k,w[j],box),
                'low_high_intrinsics_max_error':err,'visible_pixels_excluded':outside,
                'padding':box[0]<0 or box[1]<0 or box[2]>SIDE or box[3]>SIDE})
        row.update(views=views,visible_indices=[i for i,c in enumerate(counts) if c>0])
        if not row['visible_indices']:raise ValueError('All views empty; retain target as failure')
        objects.append(row)
    references=[]
    for arm in ARMS:
        for row in objects:
            for seed in SEEDS:
                folder=SHARED/'evaluations'/arm/row['uid']/str(seed)
                ss,mesh=read(folder/'ss.json'),read(folder/'mesh.json')
                if not ss['success'] or not mesh['success'] or mesh['fitted_transform']:raise ValueError('Original baseline incomplete')
                if ss['input_sample']!=row['source_sample']:raise ValueError('Baseline input differs')
                for b in [ss['coords'],mesh['mesh'],mesh['world_mesh']]:verify(b)
                references.append({'arm':arm,'uid':row['uid'],'seed':seed,
                    'ss':binding(folder/'ss.json'),'mesh':binding(folder/'mesh.json')})
    payload={'format':'hssd_input_adaptation.v1','objects':objects,'source_plan':binding(SHARED/'plan.json'),
        'source_evaluation_identity':binding(SHARED/'evaluation_identity.json'),'reference_records':references,
        'source_geometry':binding(SHARED/'scene_geometry.npz'),'input_frames':FRAMES,
        'variants':VARIANTS,'arms':ARMS,'seeds':SEEDS,'high_side':SIDE*HIGH_SCALE,
        'roi':'project known canonical unit cube; square margin 1.10; shift/clamp inside original photo, no artificial padding; never read GT surface bounds or amodal mask',
        'view_policy':'freeze nonzero visibility from original 518 masks before high-resolution rendering; synchronize all contexts',
        'scope':'same four training objects and privileged camera/mask/canonical frame; no new training, not heldout',
        'comparisons':[['full_all','full_visible'],['full_visible','crop518_visible'],['crop518_visible','crop2072_visible']],
        'latent_policy':'reuse existing SS/SLat targets; no mesh re-encoding',
        'metrics':'original canonical metrics and original four-view GT surface partitions; no fitted transform'}
    frozen={'identity':digest(payload),'plan':payload};locked(OUT/'plan.json',frozen)
    high=dict(original)
    high.update(image_side=SIDE*HIGH_SCALE,intrinsics=[(resize_affine(HIGH_SCALE)@np.array(k)).tolist() for k in original['intrinsics']],
        parent_input_protocol=frozen['identity'])
    locked(OUT/'hires/plan.json',{'identity':digest(high),'plan':high})
    result={'time':now(),'plan_identity':frozen['identity'],'passed':True,'checks':checks,
        'logical_cases':len(cases()),'new_sampling_cases':len(new_jobs()),'original_references':len(references),
        'view_sets':{o['uid']:o['visible_indices'] for o in objects}}
    if result['new_sampling_cases']!=54:raise ValueError('Unexpected case count; inspect empty-view policy')
    write(OUT/'preflight.json',result)
    event('prepared',logical_cases=result['logical_cases'],new_meshes=result['new_sampling_cases'])

if __name__=='__main__':prepare()
