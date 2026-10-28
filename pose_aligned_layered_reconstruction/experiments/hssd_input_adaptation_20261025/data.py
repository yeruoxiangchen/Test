"""Native camera audit, image-only ROI conversion, and frozen DINO caches."""
import argparse
import gc
import numpy as np
from PIL import Image,ImageDraw
from .common import *
from .geometry import crop_affine,crop_image,resize_affine


def audit_hires(frames=None):
    import torch
    import nvdiffrast.torch as dr
    from ..hssd_shared_scene_20261025.cache import first_hit
    torch.set_num_threads(2)
    p=plan()['plan'];hp=read(OUT/'hires/plan.json')
    source=np.load(verify(p['source_geometry']))
    v,f,owners=source['vertices'],source['faces'],source['owners']
    context=dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    result=[]
    for frame in FRAMES if frames is None else frames:
        receipt=read(OUT/'hires/frames'/f'{frame:03d}.json')
        if receipt['plan_identity']!=hp['identity']:raise ValueError('Wrong high-resolution frame')
        rgb=OUT/'hires/rgb'/f'{frame:03d}.png'
        with Image.open(rgb) as image:
            if image.size!=(2072,2072):raise ValueError('Wrong high-resolution raster')
        event('audit_high_mask',frame=frame)
        hit,_=first_hit(v,f,hp['plan']['intrinsics'][frame],hp['plan']['world_w2c'][frame],context,side=2072)
        ids=np.where(hit>=0,owners[hit.clip(min=0)],0)
        for obj in p['objects']:
            path=OUT/'hires/masks'/obj['uid']/f'{frame:03d}.png'
            observed=np.asarray(Image.open(path).convert('L'))>127
            expected=ids==obj['instance']+2
            union=int((observed|expected).sum())
            iou=float((observed&expected).sum()/union) if union else 1.
            result.append({'uid':obj['uid'],'frame':frame,'iou':iou,'visible_pixels':int(observed.sum()),
                'mask':binding(path),'rgb':binding(rgb)})
    passed=all(r['iou']>=.95 for r in result)
    output=OUT/('hires_audit.json' if frames is None else 'hires_preview_audit.json')
    write(output,{'time':now(),'plan_identity':plan()['identity'],'passed':passed,'rows':result})
    if not passed:raise ValueError('High-resolution RGB/mask projection audit failed')
    return result


def images(domain,allow_partial=False):
    if domain not in ['crop518','crop2072']:raise ValueError(domain)
    high=domain=='crop2072';scale=HIGH_SCALE if high else 1
    p=plan()['plan'];sheet_rows=[]
    for obj in p['objects']:
        uid=obj['uid'];folder=OUT/'inputs'/domain/uid;folder.mkdir(parents=True,exist_ok=True)
        entries=[]
        for item in obj['views']:
            frame=item['frame'];source=OUT/'hires' if high else SHARED
            rgb=source/'rgb'/f'{frame:03d}.png';mask=source/'masks'/uid/f'{frame:03d}.png'
            if allow_partial and (not rgb.exists() or not mask.exists()):continue
            if not high:verify(item['source_rgb']);verify(item['source_mask'])
            elif read(source/'frames'/f'{frame:03d}.json')['plan_identity']!=read(source/'plan.json')['identity']:
                raise ValueError('Wrong high-resolution source frame')
            box=item['box_2072'] if high else item['box_518']
            inputs={'rgb':binding(rgb),'mask':binding(mask)}
            a=crop_affine(box)
            k0=np.array(read(SHARED/'plan.json')['plan']['intrinsics'][frame])
            k=a@resize_affine(scale)@k0
            if not np.allclose(k,item['intrinsics'],atol=1.e-9):raise ValueError('Crop K mismatch')
            with Image.open(rgb) as image:raw=crop_image(image.convert('RGB'),box)
            with Image.open(mask) as image:cropped_mask=crop_image(image.convert('L'),box,mask=True)
            pixels=np.asarray(cropped_mask)>127
            # Apply the target mask after the shared RGB/mask transform, as in
            # the existing feature pipeline. Never add amodal or clean pixels.
            masked=Image.fromarray(np.asarray(raw)*pixels[:,:,None].astype(np.uint8))
            paths={'raw':folder/f'{frame:03d}_raw.png','mask':folder/f'{frame:03d}_mask.png','masked':folder/f'{frame:03d}_masked.png'}
            for key,img in [('raw',raw),('mask',cropped_mask),('masked',masked)]:img.save(paths[key])
            entry={'frame':frame,'box':box,'intrinsics':k.tolist(),'extrinsics':item['extrinsics'],
                'affine':a.tolist(),'sources':inputs,'files':{key:binding(path) for key,path in paths.items()},
                'visible_pixels_after_crop':int(pixels.sum()),
                'selected_for_model':obj['views'].index(item) in obj['visible_indices']}
            entries.append(entry)
            shown=Image.composite(raw,Image.new('RGB',raw.size,'white'),cropped_mask)
            sheet_rows.append((uid,frame,shown))
        if len(entries)==4:
            # Image generation is deterministic; changed sources cannot replace
            # the receipt of a cached model input.
            locked(folder/'images.json',{'plan_identity':plan()['identity'],'domain':domain,'uid':uid,'views':entries})
        elif not allow_partial:raise ValueError('Missing required shared frame')
    side=259
    sheet=Image.new('RGB',(4*side,4*(side+24)), 'white');draw=ImageDraw.Draw(sheet)
    for uid,frame,img in sheet_rows:
        row=next(i for i,o in enumerate(p['objects']) if o['uid']==uid);col=FRAMES.index(frame)
        x,y=col*side,row*(side+24)
        draw.text((x+4,y+4),f'{uid.split("_")[-1]} frame {frame} / {domain}',fill='black')
        sheet.paste(img.resize((side,side)),(x,y+24))
    sheet.save(OUT/f'{domain}_input_sheet.jpg',quality=95)
    event('roi_images_ready',domain=domain,object_views=len(sheet_rows))


def cache(domains):
    import torch
    from ...paired_encoder import build_frozen_dino_encoder,encode_paired_views
    from ...v2.contracts import validate_sample
    torch.set_num_threads(2);device=torch.device('cuda:0');encoder=None
    p=plan()['plan']
    hub=Path(torch.hub.get_dir())
    encoder_identity={'model':'dinov2_vitl14_reg','weights':binding(hub/'checkpoints/dinov2_vitl14_reg4_pretrain.pth'),
        'sources':[binding(path) for path in sorted((hub/'facebookresearch_dinov2_main/dinov2').rglob('*.py'))]+
            [binding(hub/'facebookresearch_dinov2_main/hubconf.py'),binding(ROOT/'ReconViaGen/trellis/pipelines/trellis_image_to_3d.py')]}
    locked(OUT/'encoder_identity.json',encoder_identity)
    encoder_bound=binding(OUT/'encoder_identity.json')
    def ensure_encoder():
        nonlocal encoder
        if encoder is None:
            event('load_dino');encoder=build_frozen_dino_encoder(device=device)
        replay=OUT/'encoder_replay.json'
        if replay.exists():
            prior=read(replay)
            if prior['encoder']!=encoder_bound or not prior['passed']:raise ValueError('Encoder replay identity changed')
            return
        # Recompute the original full-frame features to separate an encoder
        # change from an intended ROI change. Compare the saved fp16 tensors.
        from ...paired_preprocessing import make_masked_rgb
        obj=p['objects'][0];old=torch.load(verify(obj['source_sample']),map_location='cpu',weights_only=True)
        raw=[Image.open(verify(v['source_rgb'])).convert('RGB') for v in obj['views']]
        masks=[np.asarray(Image.open(verify(v['source_mask'])).convert('L'),np.float32)/255 for v in obj['views']]
        masked=[Image.fromarray(make_masked_rgb(np.asarray(img),mask)) for img,mask in zip(raw,masks)]
        event('replay_original_dino',uid=obj['uid'])
        features=encode_paired_views(encoder,raw_images=raw,masked_images=masked);checks={}
        for key in ['raw_visual_patch_features','masked_visual_patch_features']:
            diff=features[key].half().cpu().float()-old[key].float()
            checks[key]={'max_abs':float(diff.abs().max()),'rms':float(diff.square().mean().sqrt()),'exact':bool((diff==0).all())}
        passed=all(v['max_abs']<=.002 and v['rms']<=.0001 for v in checks.values())
        write(replay,{'encoder':encoder_bound,'source':obj['source_sample'],'checks':checks,'passed':passed})
        if not passed:raise ValueError('Original DINO replay differs; stop before interpreting input adaptation')
    for domain in domains:
        images(domain)
        for obj in p['objects']:
            folder=OUT/'inputs'/domain/obj['uid'];receipt=folder/'receipt.json'
            image_binding=binding(folder/'images.json')
            if receipt.exists():
                saved=read(receipt)
                if saved['images']!=image_binding or saved['plan_identity']!=plan()['identity'] or saved['encoder']!=encoder_bound:raise ValueError('Cached image/encoder identity changed')
                verify(saved['sample']);continue
            meta=read(folder/'images.json')
            for view in meta['views']:
                for b in [*view['sources'].values(),*view['files'].values()]:verify(b)
            ensure_encoder()
            event('encode_roi',domain=domain,uid=obj['uid'])
            raw=[Image.open(view['files']['raw']['path']).convert('RGB') for view in meta['views']]
            masked=[Image.open(view['files']['masked']['path']).convert('RGB') for view in meta['views']]
            with torch.no_grad():features=encode_paired_views(encoder,raw_images=raw,masked_images=masked)
            source=torch.load(verify(obj['source_sample']),map_location='cpu',weights_only=True)
            new_extrinsics=torch.tensor(np.array([v['extrinsics'] for v in meta['views']]),dtype=torch.float32)
            if not torch.equal(new_extrinsics,source['extrinsics']):raise ValueError('Crop unexpectedly changes camera extrinsics')
            # Preserve labels, world frame and source latent exactly; only
            # observation rasters/features and K change.
            source.update(uid=obj['uid']+':adapt:'+domain,
                raw_visual_patch_features=features['raw_visual_patch_features'].half().cpu(),
                masked_visual_patch_features=features['masked_visual_patch_features'].half().cpu(),
                visible_target_mask=torch.from_numpy(np.stack([np.asarray(Image.open(v['files']['mask']['path']).convert('L'),np.float32)/255 for v in meta['views']])).half(),
                intrinsics=torch.tensor(np.array([v['intrinsics'] for v in meta['views']]),dtype=torch.float32),
                extrinsics=new_extrinsics,
                source_to_feature_affines=torch.tensor(np.array([v['affine'] for v in meta['views']]),dtype=torch.float32),
                input_adaptation={'plan':plan()['identity'],'domain':domain,'images':image_binding},
                source_files=[b for v in meta['views'] for b in [*v['sources'].values(),*v['files'].values()]],
                paired_encoding_contract=features['paired_encoding_contract'])
            validate_sample(source,require_labels=False,require_ss_target=True)
            for j in obj['visible_indices']:
                if not bool(source['visible_target_mask'][j].any()):raise ValueError('Selected target disappeared during crop')
            dest=folder/'sample.pt';temp=folder/'sample.tmp';torch.save(source,temp);temp.replace(dest)
            write(receipt,{'plan_identity':plan()['identity'],'images':image_binding,'sample':binding(dest),
                'encoder':encoder_bound,
                'target_policy':'unaltered archived SS/SLat; no re-encoding'})
            del source,features
    event('cache_complete',domains=domains)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['images','cache','audit-high'])
    parser.add_argument('--domain',choices=['crop518','crop2072']);parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if args.action=='audit-high':audit_hires([0] if args.preview else None)
    elif args.action=='images':images(args.domain,args.preview)
    else:cache([args.domain] if args.domain else ['crop518','crop2072'])
