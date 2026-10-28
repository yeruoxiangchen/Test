"""Match rasterization scales when validating the post-hoc input-crop control."""
from unittest.mock import patch

from . import localization_corrected as corrected
from . import localization_crop_control as control

import numpy as np
import torch
from PIL import Image

from visual_evidence_probe.common import binding,read,write,verify,lock_json,now
from visual_evidence_probe.experiments.localization import config,runner,visuals
from visual_evidence_probe.experiments.localization.metrics import evaluate
from visual_evidence_probe.experiments.localization.report import aggregate,paired,full_summary
from visual_evidence_probe.experiments.input_consistency.refine import project_clip
from visual_evidence_probe.render import mask_iou


@torch.no_grad()
def masks_only(mesh,k,w,context):
    import nvdiffrast.torch as dr
    v=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0')
    f=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0').contiguous()
    clip=project_clip(v,k.cuda(),w.cuda())
    rast,_=dr.rasterize(context,clip,f,resolution=(518,518))
    return (rast[...,3]>0).cpu().numpy()


def main():
    import trimesh
    import nvdiffrast.torch as dr
    from visual_evidence_probe.runtime import setup
    config.check();setup();context=dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    validations=[];records=[]
    for index in range(4,12):
        rec,k,w,_,original_masks=config.observation(index)
        cr,ck,cw,images,cropped_masks=control.observation(index)
        cp=config.folder(index)/'target_canonical.ply'
        gt=trimesh.load(cp,force='mesh',process=False) if cp.exists() else runner.target_mesh(index)
        native=masks_only(gt,k,w,context)
        native_iou=mask_iou(native,original_masks)
        warped=[]
        for mask,box in zip(native,cr['crop']['crop_boxes_xyxy']):
            image=Image.fromarray(mask.astype(np.uint8)*255).crop(tuple(box)).resize((518,518),Image.Resampling.BILINEAR)
            warped.append(np.asarray(image)/255>.8)
        matched_iou=mask_iou(np.stack(warped),cropped_masks)
        # Check the declared training-coordinate affine independently of mask resampling.
        rng=np.random.default_rng(20261026)
        points=np.c_[rng.uniform(-.4,.4,(64,3)),np.ones(64)]
        maximum=0.
        for kval,new_k,camera,affine in zip(k.numpy(),ck.numpy(),w.numpy(),cr['crop']['source_to_feature_affines']):
            xyz=points@camera.T
            before=xyz[:,:3]@kval.T;before/=before[:,2:]
            after=xyz[:,:3]@new_k.T;after/=after[:,2:]
            maximum=max(maximum,float(np.abs(after-before@np.array(affine).T).max()))
        passed=bool((native_iou>=.90).all() and (matched_iou>=.90).all() and maximum<1.e-3)
        prior=read(control.folder(index)/'camera_audit.json')
        validation=dict(index=index,native_ious=native_iou.tolist(),same_resampling_ious=matched_iou.tolist(),
            cropped_mesh_raster_ious=prior['ious'],projection_affine_error_pixels=maximum,passed=passed,
            threshold=.90,validation_only_gt=True,
            reason='native rendering and crop/threshold of native rendering must match observations at the same sampling scale')
        write(control.folder(index)/'native_crop_validation.json',validation);validations.append(validation)
        print(validation,flush=True)
        if not passed:raise ValueError('like-for-like camera validation failed; no threshold relaxation')
        missing=[s for s in ('42','43') if not (control.folder(index)/f'metrics/{s}.json').exists()]
        if missing:
            b=corrected.buffers(gt,ck,cw,context,128)
            depth=b['depth'].cpu().numpy();mask=b['mask'].cpu().numpy()
            with patch.object(visuals,'folder',control.folder):
                for seed in missing:
                    path=control.folder(index)/f'scores/{seed}.npz'
                    with np.load(path) as a:maps={name:a[name] for name in a.files}
                    values={str(t):evaluate(maps,maps['pred_depth'],depth,mask,t) for t in config.PARAMS['depth_thresholds']}
                    record=dict(index=index,uid=rec['uid'],split='omni_test',key=seed,success=True,control=False,metrics=values)
                    write(control.folder(index)/f'metrics/{seed}.json',record)
                    error=(maps['pred_mask']^mask)|((maps['pred_mask']&mask)&(np.abs(maps['pred_depth']-depth)>.02))
                    meta=read(path.with_suffix('.json'));mesh=trimesh.load(verify(meta['mesh']),force='mesh',process=False)
                    visuals.visualize(index,seed,maps,depth,mask,error,images,mesh,ck,cw,context)
        records.extend(read(control.folder(index)/f'metrics/{s}.json') for s in ('42','43'))
    amendment=dict(time=now(),source=binding(__file__),base=binding(control.OUT/'protocol.json'),
        rule='retain0.90 at native and identically resampled masks, plus exact declared crop affine; report fresh crop-raster IoU separately',
        original_failure_preserved=True,score_maps_and_methods_unchanged=True,validations=validations)
    write(control.OUT/'raster_validation_amendment.json',amendment)
    summary=aggregate(records)
    comparison={f'{c}_minus_{b}':paired(summary,c,b) for c in ('direct_dino','cross_dino','fused_dino') for b in ('boundary','shuffled_dino')}
    write(control.OUT/'summary.json',dict(time=now(),records=16,summaries=summary,full_visible_summary=full_summary(records),
        paired_comparisons=comparison,all_successful=True,posthoc_control=True,parameters_unchanged=True,
        raster_validation=binding(control.OUT/'raster_validation_amendment.json')))


if __name__=='__main__':main()
