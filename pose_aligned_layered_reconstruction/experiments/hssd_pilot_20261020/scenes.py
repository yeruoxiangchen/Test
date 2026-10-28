"""Original HSSD layout -> posed clean/occluded pairs and exact layer labels."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from ...data_v2.render_scenes import load_parts, pack_scene, rasterize
from ...data_v2.build_feature_cache import cache_scene
from ...paired_encoder import build_frozen_dino_encoder
from ...v2.dataset import write_manifest
from .assets import instance_matrix, C
from .common import SOURCE, OUT, BLENDER, plan, object_dir, verify, binding, write_json, identity, event


CAMERA_SEARCH_VERSION = 'room_camera_search.v2'
CAMERA_SEARCHES = (
    {'name': 'original_60deg', 'fov_degrees': 60.,
     'distances': [1.3, 1.8, 2.5], 'angle_count': 32},
    *({'name': f'dense_{fov}deg', 'fov_degrees': float(fov),
       'distances': [.8, .95, 1.1, 1.3, 1.55, 1.8, 2.1, 2.5], 'angle_count': 64}
      for fov in (60, 75, 90)),
)


def in_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj-xi)*(y-yi)/(yj-yi) + xi:
            inside = not inside
        j = i
    return inside


def look_at(eye, center, up):
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, down, forward], axis=1)
    matrix[:3, 3] = eye
    return np.linalg.inv(matrix)


def candidates(row, mesh, *, fov_degrees=60., distances=(1.3, 1.8, 2.5),
               angle_count=32, minimum=8):
    import trimesh
    canonical = np.asarray(row['canonical_from_world'])
    world_center = np.linalg.inv(canonical)[:3, 3]
    regions = json.loads((SOURCE / 'HSSD-hab/semantics/scenes/102344049.semantic_config.json').read_text())['region_annotations']
    rooms = [r for r in regions if in_polygon(world_center[[0, 2]], np.array(r['poly_loop'])[:, [0, 2]])]
    if not rooms:
        raise ValueError('Target center has no room polygon')
    side = 518
    focal = .5 * side / math.tan(math.radians(fov_degrees) / 2)
    k = np.array([[focal, 0, (side-1)/2], [0, focal, (side-1)/2], [0, 0, 1.]])
    up = canonical[:3, :3] @ [0., 1., 0.]
    size_world = 1 / np.linalg.norm(canonical[:3, 0])
    cameras, eyes = [], []
    corners = trimesh.bounds.corners(mesh.bounds)
    for distance in distances:
        for height in [1.25, 1.65]:
            for angle in np.linspace(0, 2*np.pi, angle_count, endpoint=False):
                eye = world_center + size_world * distance * np.array([np.cos(angle), 0, np.sin(angle)])
                eye[1] = height
                if not any(in_polygon(eye[[0, 2]], np.array(r['poly_loop'])[:, [0, 2]]) for r in rooms):
                    continue
                ec = trimesh.transform_points(eye[None], canonical)[0]
                w = look_at(ec, np.zeros(3), up)
                xyz = trimesh.transform_points(corners, w)
                uv = xyz @ k.T
                uv = uv[:, :2] / uv[:, 2:]
                if xyz[:, 2].min() <= .05 or uv.min() < 8 or uv.max() > 510:
                    continue
                cameras.append(w)
                eyes.append(eye.tolist())
    if len(cameras) < minimum:
        raise ValueError(f'Only {len(cameras)} in-room non-truncated cameras for {row["uid"]}'
                         f' at FOV {fov_degrees}; need {minimum}')
    return (np.repeat(k[None], len(cameras), 0).astype(np.float32),
            np.asarray(cameras, np.float32).reshape(-1, 4, 4), eyes)


def select_input_cameras(row, mesh, assess, root):
    """Try a fixed sequence of optics, preserving every original successful pool."""
    attempts = []
    for search in CAMERA_SEARCHES:
        options = {key: value for key, value in search.items() if key != 'name'}
        k, w, eyes = candidates(row, mesh, minimum=0, **options)
        records = assess(k, w, eyes) if len(k) >= 8 else []
        eligible = [r for r in records if r['eligible']]
        record_file = root / f'camera_candidates_{search["name"]}.json'
        write_json(record_file, records)
        attempts.append({'search': search, 'geometric_candidates': len(k),
                         'eligible_candidates': len(eligible), 'records': record_file.name})
        audit = {'version': CAMERA_SEARCH_VERSION, 'uid': row['uid'], 'attempts': attempts,
                 'passed': len(eligible) >= 8}
        event(stage='camera_search', uid=row['uid'], **attempts[-1])
        if len(eligible) >= 8:
            # Each pool has distinct camera positions; do not count one position
            # repeatedly at several focal lengths as independent views.
            eligible.sort(key=lambda r: (r['visible_fraction'], r['index']))
            chosen = [eligible[i]['index']
                      for i in np.linspace(0, len(eligible)-1, 8).round().astype(int)]
            audit.update(selected_search=search['name'], selected_candidates=chosen)
            write_json(root / 'camera_search.json', audit)
            write_json(root / 'camera_candidates.json', records)
            return k[chosen], w[chosen], records, chosen, search['name']
        write_json(root / 'camera_search.json', audit)
    counts = ', '.join(f'{r["search"]["name"]}={r["eligible_candidates"]}'
                       f'/{r["geometric_candidates"]}' for r in attempts)
    raise ValueError(f'Insufficient valid input cameras for {row["uid"]}: {counts}; '
                     f'see {root / "camera_search.json"}')


def load_scene(row, device):
    canonical = np.asarray(row['canonical_from_world'])
    scene = json.loads((SOURCE / 'HSSD-hab/scenes/102344049.scene_instance.json').read_text())
    configs = {p.name.removesuffix('.object_config.json'): p
               for p in (SOURCE / 'HSSD-hab/objects').rglob('*.object_config.json')}
    converted = SOURCE / 'renders/hssd_102344049/converted_glb'
    target, others = [], load_parts(converted / 'stages/102344049.glb', canonical)
    for index, obj in enumerate(scene['object_instances']):
        cfgfile = configs[obj['template_name']]
        cfg = json.loads(cfgfile.read_text())
        path = converted / (cfgfile.parent / cfg['render_asset']).relative_to(SOURCE / 'HSSD-hab')
        rigid, deformation = instance_matrix(obj)
        parts = load_parts(path, canonical @ rigid @ deformation)
        (target if index == row['instance'] else others).extend(parts)
    # Some room materials contain one constant RGBA value rather than N vertex
    # colors. Expand that value explicitly; keep the frozen renderer unchanged.
    import trimesh
    for mesh in target + others:
        visual = mesh.visual
        material = getattr(visual, 'material', None)
        texture = getattr(material, 'image', None)
        if texture is None:
            texture = getattr(material, 'baseColorTexture', None)
        if texture is None or getattr(visual, 'uv', None) is None:
            if hasattr(visual, 'to_color'):
                colored = visual.to_color()
                colored.mesh = mesh
                rgba = np.asarray(colored.vertex_colors)
            else:
                rgba = np.asarray(visual.vertex_colors)
            if rgba.ndim == 1 and rgba.shape == (4,):
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
            elif rgba.shape != (len(mesh.vertices), 4):
                raise ValueError(f'Unsupported material color layout: {rgba.shape}')
            else:
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return pack_scene(target, others, device), pack_scene(target, [], device)


def check_layer_range(labels, w2c):
    # Existing LaRI predicts only center_depth +/- 2 canonical units. Far room
    # surfaces cannot be silently used as learnable GT. Censor whole such rays.
    depths = labels['layer_depths_gt']
    center = torch.as_tensor(w2c[:, 2, 3])[:, None, None, None]
    valid = depths > 0
    within = (~valid) | ((depths >= (center - 2).clamp_min(1.e-4)) & (depths <= center + 2))
    labels['supervision_valid_mask'] = within.all(1)
    return float(within.all(1).float().mean())


@torch.no_grad()
def build(row):
    import trimesh
    root = object_dir(row)
    audit = json.loads((root / 'latent_audit.json').read_text())
    verify(audit['ss_target']); verify(audit['slat_target'])
    mesh = trimesh.load(verify(row['mesh']), force='mesh', process=False)
    device = torch.device('cuda:0')
    event(stage='load_full_scene', uid=row['uid'])
    full, target = load_scene(row, device)
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=device)

    def assess(k, w, eyes):
        records = []
        for i in range(len(k)):
            kt = torch.from_numpy(k[i:i+1]).to(device)
            wt = torch.from_numpy(w[i:i+1]).to(device)
            clean = rasterize(target, kt, wt, raster_side=128, layers=1, context=context)
            scene = rasterize(full, kt, wt, raster_side=128, layers=1, context=context)
            area = int(clean['visible_target_mask'].sum())
            visible = int(scene['visible_target_mask'].sum())
            fraction = visible / max(area, 1)
            # Near walls/furniture indicate an invalid camera position/clearance.
            depth = scene['layer_depths_gt'][0, 0]
            clearance = float((depth > .08).float().mean())
            records.append({'index': i, 'amodal_pixels_128': area, 'visible_fraction': fraction,
                            'clearance_fraction': clearance, 'world_eye': eyes[i],
                            'eligible': area >= 80 and visible >= 24 and fraction >= .08 and clearance >= .95})
            if (i+1) % 25 == 0:
                event(stage='camera_selection', uid=row['uid'], camera=i+1, total=len(k))
        return records

    k, w, records, chosen, search_name = select_input_cameras(row, mesh, assess, root)
    kt, wt = torch.from_numpy(k).to(device), torch.from_numpy(w).to(device)
    np.savez_compressed(root / 'camera.npz', intrinsics=k, extrinsics=w)
    image_records = {}
    visible = None
    for domain, packed in [('clean', target), ('occluded', full)]:
        rendered = rasterize(packed, kt, wt, layers=1, rgb=True, context=context, background_rgb=(0., 0., 0.))
        labels = rasterize(packed, kt, wt, raster_side=37, layers=5, context=context)
        labels.pop('visible_target_mask')
        coverage = check_layer_range(labels, w)
        if coverage < .1:
            raise ValueError('Too few representable LaRI label rays')
        np.savez_compressed(root / f'{domain}_layers.npz', **{key: val.numpy() for key, val in labels.items()})
        rows = []
        for i in range(8):
            Image.fromarray((rendered['rgb'][i].numpy()*255).round().astype(np.uint8)).save(root / f'{domain}_{i:02d}.png')
            mask = rendered['visible_target_mask'][i].numpy()
            Image.fromarray((mask*255).astype(np.uint8)).save(root / f'{domain}_mask_{i:02d}.png')
            rows.append({'raw': f'{domain}_{i:02d}.png', 'visible': f'{domain}_mask_{i:02d}.png',
                         'amodal': f'clean_mask_{i:02d}.png'})
        image_records[domain] = rows
        if domain == 'clean':
            amodal = rendered['visible_target_mask'].sum((1, 2)).float()
        else:
            visible = (rendered['visible_target_mask'].sum((1, 2)) / amodal.clamp_min(1)).tolist()
        recipe = {'scene': row['scene'], 'instance': row['instance'], 'cameras': chosen,
                  'camera_search_version': CAMERA_SEARCH_VERSION, 'camera_search': search_name,
                  'domain': domain, 'source_plan': plan()['identity'], 'layer_range': 'center +/-2; outside rays censored'}
        meta = {'format': 'posed_scene_source.v2', 'uid': row['uid'] + ':' + domain,
                'object_uid': row['uid'], 'scene_uid': row['scene'], 'trajectory_uid': row['uid'] + ':views',
                'data_domain': 'full_render', 'views': rows, 'camera': 'camera.npz',
                'layers': f'{domain}_layers.npz', 'camera_forward_sign': 1., 'extrinsics_type': 'w2c',
                'layer_camera_y_sign': 1., 'depth_kind': 'camera_forward_z', 'label_geometry_kind': 'source_mesh',
                'target_mesh': 'target_canonical.ply',
                'object_from_asset': (np.array(row['normalization']) @ C @ np.array(row['deformation'])).tolist(),
                'canonical_from_world': row['canonical_from_world'],
                'legacy_source_binding': {'slat': str(root / 'slat.npz'), 'slat_sha256': binding(root / 'slat.npz')['sha256']},
                'recipe': recipe, 'recipe_hash': identity(recipe), 'crop_kind': 'full_frame_no_gt_bbox',
                'renderer': 'nvdiffrast original full HSSD geometry; opaque texture RGB and labels',
                'supervision_ray_fraction': coverage, 'alpha_policy': 'opaque_rgb_and_geometry'}
        write_json(root / f'{domain}_scene.json', meta)
    del full, target, rendered, labels
    torch.cuda.empty_cache()
    cycles = root / 'input_cycles'
    if not (cycles / 'render_receipt.json').exists():
        event(stage='cycles_input_render', uid=row['uid'], log=str(root / 'input_cycles.log'))
        with (root / 'input_cycles.log').open('a') as log:
            subprocess.run([BLENDER, '-b', '-t', '4', '--python-exit-code', '1', '-P',
                str(Path(__file__).with_name('render_inputs.py')), '--', '--asset', str(root / 'asset.json')],
                stdout=log, stderr=subprocess.STDOUT, check=True)
    pixel_checks = []
    for domain in ['clean', 'occluded']:
        meta_file = root / f'{domain}_scene.json'
        meta = json.loads(meta_file.read_text())
        for i, view in enumerate(meta['views']):
            rgba = np.asarray(Image.open(cycles / f'{domain}_{i:02d}_rgba.png').convert('RGBA'), np.float32) / 255.
            Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:] * 255).round().astype(np.uint8)).save(cycles / f'{domain}_{i:02d}.png')
            expected = np.asarray(Image.open(root / view['visible'])) > 127
            observed = np.asarray(Image.open(cycles / f'{domain}_{i:02d}_mask.png')) > 127
            iou = float((expected & observed).sum() / max((expected | observed).sum(), 1))
            pixel_checks.append({'domain': domain, 'view': i, 'silhouette_iou': iou})
            if iou < .95:
                write_json(root / 'input_projection_audit.json', {'passed': False, 'views': pixel_checks})
                raise ValueError('Cycles input and geometric label masks disagree')
            view['raw'] = f'input_cycles/{domain}_{i:02d}.png'
        meta['renderer'] = 'Cycles RGB + nvdiffrast exact depth peeling, independently verified object-ID masks'
        meta['recipe']['rgb_renderer'] = 'original scene Cycles32; opaque surfaces'
        meta['recipe_hash'] = identity(meta['recipe'])
        write_json(meta_file, meta)
    write_json(root / 'input_projection_audit.json', {'passed': True, 'views': pixel_checks})
    event(stage='load_input_dino', uid=row['uid'])
    dino = build_frozen_dino_encoder(device=device)
    rows = {}
    for domain in ['clean', 'occluded']:
        event(stage='cache_input_dino', uid=row['uid'], domain=domain)
        rows[domain] = cache_scene(root / f'{domain}_scene.json', root / 'ss.npz',
                                  root / f'{domain}_sample.pt', dino, resume=True)
    write_json(root / 'scene_receipt.json', {'rows': rows, 'visible_fraction': visible,
        'selected_candidates': chosen, 'camera_count': len(records),
        'camera_search_version': CAMERA_SEARCH_VERSION, 'camera_search': search_name,
        'scope': 'one view group per object, not independent scene/asset generalization',
        'artifacts': [binding(root / f'{domain}_scene.json') for domain in rows] + [binding(root / 'camera.npz')]})
    event(stage='scene_complete', uid=row['uid'], visible_fraction=visible)


def manifests():
    rows = {'clean': [], 'occluded': []}
    frozen = plan()
    for row in frozen['plan']['objects']:
        receipt = json.loads((object_dir(row) / 'scene_receipt.json').read_text())
        for domain in rows:
            verify({'path': receipt['rows'][domain]['sample'], 'sha256': receipt['rows'][domain]['sha256']})
            rows[domain].append(receipt['rows'][domain])
    rows['paired'] = rows['clean'] + rows['occluded']
    for domain, samples in rows.items():
        path = OUT / f'{domain}16.json'
        if not path.exists():
            write_manifest(path, samples, {'pilot_plan': frozen['identity'], 'domain': domain,
                'cohort': 'same 16 training objects; not a test set'}, OUT)
    event(stage='manifests_complete', objects=16, paired_samples=32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int)
    parser.add_argument('--manifests', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.manifests:
        manifests()
    else:
        build(plan()['plan']['objects'][args.index])
