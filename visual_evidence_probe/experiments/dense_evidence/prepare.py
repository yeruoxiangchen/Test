"""Freeze input-only cohort selection and copy only needed legacy inputs."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .config import STORE, OUT, ROUND6, LOCALIZATION, OMNI, PARAMS, data_folder, setup, read, write, digest, now, register


def physical(path):
    return Path(str(path).replace('/home/zjr/Tracker/visual_evidence_probe/outputs/', str(STORE / 'outputs') + '/'))


def freeze_registry():
    manifest = read(OMNI / 'manifest.json')
    dev = []
    for index in PARAMS['dev_indices']:
        source = LOCALIZATION / 'input_crop_resolution_control/objects' / f'{index:03d}' / 'observations.json'
        rec = read(source)
        dev.append(dict(index=index, uid=rec['uid'], split='development', kind='synthetic_clean',
            observations=str(source), source_round6=str(ROUND6 / 'objects' / f'{index:03d}')))
    excluded = {row['uid'] for row in dev}
    categories = sorted({row['category'] for row in manifest['objects']})[:PARAMS['validation_object_count']]
    rows = [min((r for r in manifest['objects'] if r['category'] == category and r['uid'] not in excluded), key=lambda r:r['uid']) for category in categories]
    val = [dict(index=index, uid=row['uid'], split='validation', kind='synthetic_clean', source=row)
        for index, row in zip(PARAMS['validation_indices'], rows)]
    original = Path('/home/zjr/Tracker/pose_aligned_layered_reconstruction/outputs/ycbv_drill_occlusion_20260910')
    real_manifest = read(original / 'manifest.json')
    real = [dict(index=index, uid=f'ycbv_drill_000056_{name}', split='real_pilot', kind='real_scene',
        group=name, legacy_root=str(original), normalization=real_manifest['normalization'],
        input_assistance=real_manifest['assistance'], rows=real_manifest['groups'][name]['rows'])
        for index, name in zip(PARAMS['real_indices'], ('A_light_8','B_occluded_2_of_8','C_occluded_8_of_8'))]
    result = dict(time=now(), selection='first eligible UID per first16 lexicographic categories, excluding all8 development UIDs',
        source_manifest=str(OMNI / 'manifest.json'), rows=dev+val+real,
        validation_outputs_or_gt_inspected_at_selection=False)
    path = OUT / 'registry.json'
    if path.exists():
        before = read(path)
        if before['rows'] != result['rows']: raise ValueError('frozen object selection changed')
        return before
    write(path, result)
    return result


def prepare_row(row):
    root = data_folder(row['index']); root.mkdir(parents=True, exist_ok=True)
    if (root / 'observations.json').exists():
        if row['kind']=='real_scene':prepare_real_domain(row)
        return
    if row['split'] == 'development':
        rec = read(row['observations'])
        for name in ('images', 'masks'):
            rec[name] = [dict(path=str(physical(value['path'])), sha256=value.get('sha256')) for value in rec[name]]
        rec.update(split=row['split'], kind=row['kind'], index=row['index'])
        write(root / 'observations.json', rec)
        write(root / 'evaluation_target.json', dict(mesh=str(LOCALIZATION / 'objects' / f'{row["index"]:03d}' / 'target_canonical.ply'), transform=np.eye(4).tolist()))
        return
    images, masks, raw_images = [], [], []
    if row['kind'] == 'synthetic_clean':
        from ar_ss_flow.shared_object_preprocessing import prepare_shared_object_arrays
        source = row['source']
        rgb = [np.asarray(Image.open(path).convert('RGB').resize((518,518), Image.Resampling.BICUBIC)) for path in source['rgb_white_images']]
        mask = [np.asarray(Image.open(path).convert('L').resize((518,518), Image.Resampling.NEAREST)) for path in source['mask_images']]
        prepared = prepare_shared_object_arrays(rgb, mask, resolution=518, foreground_margin=1.1, alpha_threshold=.8)
        resize = np.array([[518/512,0,(518/512-1)/2],[0,518/512,(518/512-1)/2],[0,0,1]],np.float32)
        original_k = resize @ np.asarray(source['intrinsic'],np.float32)
        k = prepared.source_to_feature_affines @ original_k
        w = np.linalg.inv(np.asarray(source['c2w_opencv_model_o'],np.float64)).astype(np.float32)
        for view in range(4):
            prepared.images[view].save(root / f'masked_{view:02d}.png')
            # Exact cropped foreground; DA3 primary uses the same black background.
            images.append(dict(path=str(root / f'masked_{view:02d}.png')))
            # The shared contract returns thresholded foreground masks.
            alpha = np.asarray(prepared.masks[view])
            Image.fromarray(np.uint8(alpha > .5)*255).save(root / f'mask_{view:02d}.png')
            masks.append(dict(path=str(root / f'mask_{view:02d}.png')))
        target = dict(mesh=source['source_mesh'], transform=source['source_to_model_o_4x4'])
        crop = prepared.geometry_record()
    else:
        source_root = Path(row['legacy_root']) / row['group']
        for view in range(8):
            for source_name, output, collection in ((f'masked_{view:02d}.png',f'masked_{view:02d}.png',images),
                    (f'raw_{view:02d}.png',f'raw_{view:02d}.png',raw_images),
                    (f'mask_{view:02d}.png',f'mask_{view:02d}.png',masks)):
                shutil.copyfile(source_root / source_name, root / output)
                collection.append(dict(path=str(root / output)))
        for name in ('geometry.npz','sample.pt'): shutil.copyfile(source_root/name, root/name)
        with np.load(root / 'geometry.npz') as a: k, w = a['intrinsics'], a['extrinsics']
        # Copy the already canonicalized GT once per group; it remains evaluation-only.
        shutil.copyfile(Path(row['legacy_root'])/'gt_drill_canonical.ply', root/'target_canonical.ply')
        target = dict(mesh=str(root/'target_canonical.ply'), transform=np.eye(4).tolist())
        crop = dict(source='inherited annotated BOP crop', assistance=row['input_assistance'])
    for collection in (images,masks,raw_images):
        for entry in collection: entry['sha256']=digest(entry['path'])
    rec = dict(index=row['index'], uid=row['uid'], split=row['split'], kind=row['kind'],
        intrinsics=np.asarray(k).tolist(), extrinsics=np.asarray(w).tolist(), images=images, masks=masks,
        raw_images=raw_images, crop=crop, input_uses_depth=False)
    write(root/'evaluation_target.json',target)
    write(root/'observations.json',rec)
    if row['kind']=='real_scene':prepare_real_domain(row)


def prepare_real_domain(row):
    root=data_folder(row['index']);path=root/'observation_domain.npz'
    rec=read(root/'observations.json')
    if not path.exists():
        with np.load(root/'geometry.npz') as a:affines=a['affines']
        yy,xx=np.mgrid[:518,:518];valid=[]
        for i,source in enumerate(row['rows']):
            x,y,width,height=source['bbox_obj']
            corners=np.array([[x,y,1],[x+width,y+height,1]],np.float32)@affines[i].T
            visible=np.asarray(Image.open(rec['masks'][i]['path']).convert('L'))>127
            negative=(xx<corners[0,0]-2)|(xx>corners[1,0]+2)|(yy<corners[0,1]-2)|(yy>corners[1,1]+2)
            valid.append(visible|negative)
        np.savez_compressed(path,valid=np.stack(valid))
    rec['observation_domain']=str(path)
    rec['observation_domain_assistance']='visible target mask plus exterior of inherited annotated full-object bounding rectangle'
    write(root/'observations.json',rec)


def main():
    setup()
    from pose_aligned_layered_reconstruction._bootstrap import bootstrap_external_paths
    bootstrap_external_paths()
    register('prepare')
    rows = freeze_registry()['rows']
    for row in rows:
        prepare_row(row)
        print(dict(stage='input_prepared',index=row['index'],uid=row['uid'],split=row['split']),flush=True)


if __name__ == '__main__': main()
