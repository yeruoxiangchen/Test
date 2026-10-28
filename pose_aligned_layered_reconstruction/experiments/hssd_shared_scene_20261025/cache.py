"""Independent camera/mask check, shared-view caches, and evaluation-only GT."""
import numpy as np
from PIL import Image, ImageDraw
import torch
from .common import *
from .geometry import canonical_extrinsics, contract_checks
from ...v2.contracts import SAMPLE_FORMAT, validate_sample
from ...v2.dataset import write_manifest
from ...paired_encoder import build_frozen_dino_encoder, encode_paired_views
from ...paired_preprocessing import make_masked_rgb


@torch.no_grad()
def first_hit(vertices, faces, k, w, context, side=518):
    import nvdiffrast.torch as dr
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device='cuda:0')
    faces = torch.as_tensor(faces, dtype=torch.int32, device='cuda:0').contiguous()
    k, w = vertices.new_tensor(k), vertices.new_tensor(w)
    points = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1) @ w.T
    z = points[:, 2]
    x = 2 * (k[0, 0] * points[:, 0] + (k[0, 2] + .5) * z) / side - z
    y = 2 * (k[1, 1] * points[:, 1] + (k[1, 2] + .5) * z) / side - z
    near, far = .001, 200.
    clip = torch.stack([x, y, (far+near)/(far-near)*z - 2*far*near/(far-near), z], dim=-1)[None].contiguous()
    rast, _ = dr.rasterize(context, clip, faces, (side, side))
    depth, _ = dr.interpolate(z[None, :, None].contiguous(), rast, faces)
    indices = rast[0, ..., 3].long() - 1
    return indices.cpu().numpy(), torch.where(indices >= 0, depth[0, ..., 0], 0.).cpu().numpy()


@torch.no_grad()
def audit_geometry():
    import nvdiffrast.torch as dr
    frozen = plan()
    p = frozen['plan']
    for i in FRAMES:
        if read(OUT / 'frames' / f'{i:03d}.json')['plan_identity'] != frozen['identity']:
            raise ValueError('Missing or changed shared frame')
    data = np.load(OUT / 'scene_geometry.npz')
    vertices, faces, owners = data['vertices'], data['faces'], data['owners']
    import trimesh
    from scipy.spatial import cKDTree
    mesh_checks=[]
    for row in p['objects']:
        target=trimesh.load(verify(row['mesh']),force='mesh',process=False)
        target.apply_transform(np.linalg.inv(row['canonical_from_world']))
        scene_points=vertices[np.unique(faces[owners==row['instance']+2])]
        # Compare the actual archived GT and Blender object, not just an
        # algebraically self-consistent camera matrix round trip.
        forward=cKDTree(scene_points).query(target.vertices,workers=2)[0]
        backward=cKDTree(target.vertices).query(scene_points,workers=2)[0]
        error=max(float(forward.max()),float(backward.max()))
        mesh_checks.append({'uid':row['uid'],'symmetric_vertex_max_distance_world':error,
                            'tolerance_world':1.e-4,'passed':error<=1.e-4})
    context = dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    records, view_labels = [], {}
    for i in FRAMES:
        event('geometry_audit', frame=i)
        k, w = p['intrinsics'][i], p['world_w2c'][i]
        hit, depth = first_hit(vertices, faces, k, w, context)
        ids = np.where(hit >= 0, owners[hit.clip(min=0)], 0)
        # Shared observed depth is kept under evaluation/, never in model inputs.
        target_folder = OUT / 'evaluation_gt'
        target_folder.mkdir(exist_ok=True)
        np.savez_compressed(target_folder / f'{i:03d}.npz', depth_z_world=depth, instance_id=ids)
        for row in p['objects']:
            visible = np.asarray(Image.open(OUT / 'masks' / row['uid'] / f'{i:03d}.png').convert('L')) > 127
            geometric_visible = ids == row['instance'] + 2
            intersection, union = int((visible & geometric_visible).sum()), int((visible | geometric_visible).sum())
            iou = intersection / union if union else 1.
            target_faces = faces[owners == row['instance'] + 2]
            amodal_hit, target_depth = first_hit(vertices, target_faces, k, w, context)
            amodal = amodal_hit >= 0
            label_folder = target_folder / row['uid']
            label_folder.mkdir(exist_ok=True)
            Image.fromarray(amodal.astype(np.uint8) * 255).save(label_folder / f'{i:03d}_amodal.png')
            np.savez_compressed(label_folder / f'{i:03d}.npz', amodal=amodal,
                                visible=geometric_visible, target_depth_z_world=target_depth)
            record = {'uid': row['uid'], 'frame': i, 'mask_iou': iou,
                'visible_pixels': int(visible.sum()), 'amodal_pixels': int(amodal.sum()),
                'visible_fraction': float(geometric_visible.sum() / max(amodal.sum(), 1)),
                'camera_clearance_fraction': float(((depth == 0) | (depth > .08)).mean())}
            records.append(record)
            view_labels[row['uid'], i] = amodal
    # Geometry agreement is an implementation check, not target difficulty selection.
    passed = all(r['mask_iou'] >= .95 and r['camera_clearance_fraction'] >= .95 for r in records)
    passed &= all(sum(r['visible_pixels'] for r in records if r['uid'] == obj['uid']) > 0 for obj in p['objects'])
    passed &= all(r['passed'] for r in mesh_checks)
    result = {'time': now(), 'plan_identity': frozen['identity'], 'passed': bool(passed),
              'views': records, 'projection_checks': contract_checks(p['objects']),
              'archived_gt_to_blender_world':mesh_checks,
              'scope': 'GT is used only for this calibrated diagnostic and scoring; no view/target discarded'}
    write(OUT / 'geometry_audit.json', result)
    if not passed:
        raise ValueError('Shared-camera/mask audit failed; see geometry_audit.json. No object or frame was dropped.')
    return result


def galleries(p):
    side = 259
    sheet = Image.new('RGB', (side * len(FRAMES), side * (1 + len(p['objects'])) + 30), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, frame in enumerate(FRAMES):
        rgb = Image.open(OUT / 'rgb' / f'{frame:03d}.png').convert('RGB')
        sheet.paste(rgb.resize((side, side)), (col*side, 30))
        draw.text((col*side+6, 5), f'Shared frame {frame}', fill='black')
        for j, row in enumerate(p['objects']):
            mask = Image.open(OUT / 'masks' / row['uid'] / f'{frame:03d}.png').convert('L')
            selected = Image.composite(rgb, Image.new('RGB', rgb.size, 'white'), mask)
            sheet.paste(selected.resize((side, side)), (col*side, 30+(j+1)*side))
            draw.text((col*side+5, 35+(j+1)*side), row['uid'].split('_')[-1]+' '+row['category'], fill='red')
    sheet.save(OUT / 'shared_input_contact_sheet.jpg', quality=95)


@torch.no_grad()
def cache_inputs():
    torch.set_num_threads(2)
    frozen, rows = plan(), []
    p = frozen['plan']
    audit_geometry()
    galleries(p)
    # Existing labelled latent archives are bound for scoring/possible later
    # fitting only. Inference uses the old explicit observation allowlist.
    device = torch.device('cuda:0')
    encoder = None
    for row in p['objects']:
        folder = OUT / 'objects' / row['uid']
        folder.mkdir(parents=True, exist_ok=True)
        dest, receipt = folder / 'sample.pt', folder / 'receipt.json'
        source_paths = [OUT/'rgb'/f'{i:03d}.png' for i in FRAMES] + [OUT/'masks'/row['uid']/f'{i:03d}.png' for i in FRAMES]
        source_bindings = [binding(x) for x in source_paths]
        if receipt.exists():
            old = read(receipt)
            if old['plan_identity'] != frozen['identity'] or old['source_files'] != source_bindings:
                raise ValueError('Changed cached observations')
            verify({'path': old['row']['sample'], 'sha256': old['row']['sha256']})
            rows.append(old['row'])
            continue
        if encoder is None:
            event('load_dino')
            encoder = build_frozen_dino_encoder(device=device)
        event('cache_features', uid=row['uid'])
        raw = [Image.open(OUT/'rgb'/f'{i:03d}.png').convert('RGB') for i in FRAMES]
        masks = [np.asarray(Image.open(OUT/'masks'/row['uid']/f'{i:03d}.png').convert('L'), np.float32)/255 for i in FRAMES]
        masked = [Image.fromarray(make_masked_rgb(np.asarray(img), mask)) for img,mask in zip(raw,masks)]
        features = encode_paired_views(encoder, raw_images=raw, masked_images=masked)
        with np.load(verify(row['ss'])) as archive:
            ss_key = 'target' if 'target' in archive else 'z'
            target = torch.from_numpy(np.asarray(archive[ss_key], np.float32).reshape(8,16,16,16).copy())
            coords = torch.from_numpy(archive['official_coords'].astype(np.int32))
        w = canonical_extrinsics(np.asarray(p['world_w2c'])[FRAMES], row['canonical_from_world'])
        sample = {'format': SAMPLE_FORMAT, 'uid': row['uid']+':shared_dining_v1', 'object_uid': row['uid'],
            'scene_uid': p['scene'], 'trajectory_uid': p['trajectory_uid'], 'data_domain': 'full_render',
            'depth_kind': 'camera_forward_z', 'image_size': (518,518), 'grid_transform': 'identity',
            'intrinsics': torch.tensor(np.asarray(p['intrinsics'])[FRAMES],dtype=torch.float32),
            'extrinsics': torch.tensor(w,dtype=torch.float32), 'extrinsics_type': 'w2c',
            'camera_forward_sign': 1., 'layer_camera_y_sign': 1.,
            'raw_visual_patch_features': features['raw_visual_patch_features'].half().cpu(),
            'masked_visual_patch_features': features['masked_visual_patch_features'].half().cpu(),
            'visible_target_mask': torch.from_numpy(np.stack(masks)).half(),
            'target': target, 'target_coords': coords, 'target_mesh_binding': row['mesh'],
            'source_binding': {'slat': row['slat']['path'], 'slat_sha256': row['slat']['sha256']},
            'ss_target_binding': row['ss'], 'source_files': source_bindings,
            'canonical_from_world': row['canonical_from_world'], 'shared_frame_ids': FRAMES,
            'privileged_input_metadata': ['rendered_camera', 'visible_instance_mask', 'GT_canonical_frame'],
            'paired_encoding_contract': features['paired_encoding_contract']}
        validate_sample(sample, require_labels=False, require_ss_target=True)
        temp = dest.with_suffix('.tmp')
        torch.save(sample, temp); temp.replace(dest)
        manifest_row = {k:sample[k] for k in ['uid','object_uid','scene_uid']}
        manifest_row.update(sample=str(dest),sha256=sha(dest))
        write(receipt, {'plan_identity':frozen['identity'],'source_files':source_bindings,'row':manifest_row})
        rows.append(manifest_row)
    config = {'shared_plan':frozen['identity'],'scope':p['scope'],'conditioning':p['conditioning'],'input_frames':FRAMES}
    manifest = OUT/'manifest.json'
    if not manifest.exists():
        write_manifest(manifest,rows,config,OUT)
    elif read(manifest)['samples'] != rows:
        raise ValueError('Changed completed shared manifest')
    event('data_complete', objects=len(rows), common_views=len(FRAMES))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.audit_only:
        audit_geometry()
        galleries(plan()['plan'])
    else:
        cache_inputs()
