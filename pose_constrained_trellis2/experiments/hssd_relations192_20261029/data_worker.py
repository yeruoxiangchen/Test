"""Input-only grouping, native-codec reuse, and shared-frame DINO conditions."""
import argparse
import copy
import os
from pathlib import Path
from . import common as c


def view_record(scene, uid, frame):
    root = c.SOURCE / 'scenes' / scene
    return {'frame': int(frame), 'rgb': {'path': str(root / 'rgb' / f'{frame:03d}.png')},
            'mask': {'path': str(root / 'masks' / uid / f'{frame:03d}.png')}}


def select_scene(scene):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation
    plan = c.plan()
    anchors = [r for r in plan['anchors'] if r['scene'] == scene]
    candidates = [r for r in plan['eligible_candidates'] if r['scene'] == scene]
    folder = c.SOURCE / 'scenes' / scene
    capture = c.read(folder / 'plan.json')['plan']
    measurements = {a['uid']: {} for a in anchors}
    frames = sorted({v['frame'] for a in anchors for v in a['views']})
    for frame in frames:
        masks, areas, grown = {}, {}, {}
        for row in candidates:
            with Image.open(folder / 'masks' / row['uid'] / f'{frame:03d}.png') as im:
                mask = np.asarray(im.resize((518, 518), Image.Resampling.NEAREST)) > 127
            area = int(mask.sum())
            if area >= 128:
                masks[row['uid']] = mask; areas[row['uid']] = area
        for anchor in anchors:
            uid = anchor['uid']
            if frame not in [v['frame'] for v in anchor['views']] or uid not in masks:
                continue
            grown[uid] = binary_dilation(masks[uid], iterations=5)
            for neighbor in candidates:
                other = neighbor['uid']
                if uid == other or other not in masks: continue
                touching = int((grown[uid] & masks[other]).sum())
                measurements[uid].setdefault(other, []).append(
                    {'frame': frame, 'score': touching / min(areas[uid], areas[other]),
                     'pixels': [areas[uid], areas[other]]})
        print(f'SCENE_MASKS {scene} frame={frame}', flush=True)
    groups = []
    eyes = np.asarray(capture['world_eye'])
    for anchor in anchors:
        choices = []
        for uid, rows in measurements[anchor['uid']].items():
            selected = []
            for item in sorted(rows, key=lambda x: (-x['score'], -min(x['pixels']), x['frame'])):
                if any(np.linalg.norm(eyes[item['frame']] - eyes[v['frame']]) < .2 for v in selected):
                    continue
                selected.append(item)
            if len(selected) >= 2:
                choices.append((uid, selected))
        choices.sort(key=lambda x: (-max(r['score'] for r in x[1]),
                                    -sum(min(r['pixels']) for r in x[1]), x[0]))
        if choices:
            neighbor, chosen = choices[0]
            uids, selected_frames = [anchor['uid'], neighbor], [r['frame'] for r in chosen]
        else:
            # Keep every original anchor, including those without a co-visible neighbor.
            uids, selected_frames = [anchor['uid']], [r['frame'] for r in anchor['views']]
        groups.append({'id': anchor['uid'], 'anchor': anchor['uid'], 'scene': scene,
                       'split': anchor['split'], 'uids': uids, 'frames': selected_frames,
                       'evidence': 'visible mask boundary proximity in frozen anchor views; not GT depth order',
                       'singleton': len(uids) == 1})
    c.immutable(c.DATA / 'scene_groups' / f'{scene}.json', {'groups': groups})


def assemble():
    plan = c.plan()
    groups = [r for scene in sorted({r['scene'] for r in plan['anchors']})
              for r in c.read(c.DATA / 'scene_groups' / f'{scene}.json')['groups']]
    assert len(groups) == 192 and {g['anchor'] for g in groups} == {r['uid'] for r in plan['anchors']}
    pilot = [{'id': f'pilot_{i:02d}', 'anchor': r['a'], 'scene': r['scene'], 'split': r['split'],
              'uids': [r['a'], r['b']], 'frames': [v['frame'] for v in r['selected_frames']],
              'evidence': 'previously frozen input-only pair; GT diagnostic never read'}
             for i, r in enumerate(plan['pilot_groups'])]
    wanted = {u for g in groups + pilot for u in g['uids']}
    rows = [r for r in plan['eligible_candidates'] if r['uid'] in wanted]
    assert len(rows) == len(wanted)
    for split in ('train', 'val', 'test'):
        own = [r for r in rows if r['split'] == split]
        other = [r for r in rows if r['split'] != split]
        for field in ('scene', 'asset_family', 'candidate_geometry_hash'):
            assert not ({r[field] for r in own} & {r[field] for r in other}), field
    c.immutable(c.DATA / 'groups.json', {'groups': groups, 'pilot_groups': pilot})
    c.immutable(c.DATA / 'objects.json', {'objects': rows})
    print({'objects_to_encode_or_reuse': len(rows), 'anchors': 192,
           'paired_groups': sum(len(g['uids']) == 2 for g in groups)}, flush=True)


def validate_reuse(uid):
    reuse = c.plan()['native_reuse'][uid]
    folder = Path(reuse['folder'])
    for name, size in reuse['files'].items():
        assert (folder / name).stat().st_size == size
    source = next(r for r in c.read(c.DATA / 'objects.json')['objects'] if r['uid'] == uid)
    assert source['mesh'] == reuse['source_mesh']
    for name in ('shape_512.npz', 'shape_1024.npz', 'ss.npz'):
        assert (folder / name).is_file()
    c.write(c.OUT / 'reuse' / f'{uid}.json', {'uid': uid, 'read_only_folder': str(folder), 'source': reuse})


def codec(action, uid, resolution):
    """Reuse frozen native implementation in this process with explicit local paths."""
    from types import SimpleNamespace
    from ..hssd_scene_20261028 import worker
    if uid in c.plan()['native_reuse']:
        raise ValueError('Never write reused Native12 objects')
    rows = c.read(c.DATA / 'objects.json')['objects']
    row = next(r for r in rows if r['uid'] == uid)
    if action == 'dual' and resolution == 512:
        assert c.sha(row['mesh']['path']) == row['mesh']['sha256'], 'Source mesh changed'
    worker.c = SimpleNamespace(DATA=c.DATA, OUT=c.OUT, models=c.models, read=c.read,
                              write=c.write, immutable=c.immutable, now=c.now,
                              plan=lambda: {'plan': {'objects': rows}})
    if action == 'encode': encode_native(uid, resolution)
    elif action == 'dual': worker.dual(uid, resolution)
    else: getattr(worker, action)(uid)


def encode_native(uid, resolution):
    """Unchanged official encoder math; validate/save the complete native support."""
    import numpy as np
    import torch, o_voxel
    from trellis2 import models
    from trellis2.modules import sparse as sp
    from ...scene_relations.shape_support import validate_shape
    assert resolution in (512, 1024)
    out = c.codec_folder(uid)
    coords, attr = o_voxel.io.read_vxz(str(out / f'dual_{resolution}.vxz'), num_threads=2)
    vertices = sp.SparseTensor((attr['vertices'] / 255.).float(), torch.cat([torch.zeros_like(coords[:, :1]), coords], -1))
    bits = attr['intersected']
    inter = vertices.replace(torch.cat([bits % 2, bits // 2 % 2, bits // 4 % 2], -1).bool())
    encoder = models.from_pretrained(c.models()['shape_encoder']).eval().cuda()
    with torch.no_grad(): z = encoder(vertices.cuda(), inter.cuda(), sample_posterior=False)
    xyz = z.coords[:, 1:].cpu().numpy().astype(np.int32)
    feats = z.feats.float().cpu().numpy()
    maximum = c.CONFIG['shape_max_tokens'] if resolution == 512 else (resolution // 16) ** 3
    validate_shape(xyz, feats, resolution=resolution // 16, max_tokens=maximum)
    np.savez_compressed(out / f'shape_{resolution}.npz', coords=xyz, feats=feats)
    c.write(out / f'encode_{resolution}.json', {'resolution': resolution, 'tokens': len(xyz), 'channels': feats.shape[1],
            'complete_native_support': True, 'truncated': False, 'runtime_identity': c.identity()['identity']})


def group_for(gid):
    d = c.read(c.DATA / 'groups.json')
    return next(g for g in d['groups'] + d['pilot_groups'] if g['id'] == gid)


def conditions(gid):
    import numpy as np
    import torch
    from ..hssd_scene_20261028.images import prepare_view
    from ..hssd_scene_20261028.dino import NativeDinoV3FeatureExtractor
    from ...constants import SAMPLE_FORMAT
    group = group_for(gid)
    capture = c.read(c.SOURCE / 'scenes' / group['scene'] / 'plan.json')['plan']
    assets = {r['uid']: r for r in c.read(c.DATA / 'objects.json')['objects']}
    extractor = NativeDinoV3FeatureExtractor(c.models()['dino'], image_size=512)
    extractor.to('cuda')
    for uid in group['uids']:
        out = c.DATA / 'conditions' / gid / uid; out.mkdir(parents=True, exist_ok=True)
        features, intrinsics, extrinsics, metadata = [], [], [], []
        views = [view_record(group['scene'], uid, frame) for frame in group['frames']]
        for view in views:
            frame = view['frame']
            image, K, meta = prepare_view(Path(view['rgb']['path']), Path(view['mask']['path']),
                torch.tensor(capture['intrinsics'][frame], dtype=torch.float32), margin=.1, threshold=.8)
            image.save(out / f'input_{frame:03d}.png')
            with torch.no_grad(): feature = extractor([image]).detach().half().cpu()
            features.append(feature); intrinsics.append(K); extrinsics.append(capture['world_w2c'][frame]); metadata.append(meta)
        feature = torch.cat(features)
        assert tuple(feature.shape[1:]) == (1029, 1024) and torch.isfinite(feature).all()
        transform = c.read(c.codec_folder(uid) / 'transform.json')['t2_to_world']
        payload = {'format': SAMPLE_FORMAT, 'uid': uid, 'dino_features': feature,
                   'intrinsics': torch.stack(intrinsics), 'extrinsics': torch.tensor(extrinsics, dtype=torch.float32),
                   'extrinsics_type': 'w2c', 'camera_forward_sign': 1., 'camera_image_y_sign': 1.,
                   'grid_transform': 'identity', 'image_size': [512, 512],
                   'object_to_world': torch.tensor(transform, dtype=torch.float32),
                   'frame_ids': group['frames'], 'scene_id': group['scene'], 'instance_id': assets[uid]['instance'],
                   'source_images': [v['rgb']['path'] for v in views], 'source_masks': [v['mask']['path'] for v in views],
                   'view_preprocessing': metadata, 'pose_source': c.CONFIG['pose_source']}
        c.atomic_torch_save(payload, out / 'pose.pt')
        hits = projection_audit(group, uid, capture)
        c.write(out / 'condition.json', {'uid': uid, 'group': gid, 'frame_ids': group['frames'],
                                      'features': list(feature.shape), 'pose_sha256': c.sha(out / 'pose.pt'),
                                      'gt_projection_diagnostic_only': hits})
        if min(r['visible_to_amodal_hit'] for r in hits) < .75:
            raise RuntimeError(f'Input/GT coordinate audit failed: {gid}/{uid}; inspect condition.json, do not drop target')
        print(f'CONDITIONS {gid} {uid}', flush=True)


def projection_audit(group, uid, capture):
    import numpy as np
    import torch, trimesh
    from PIL import Image
    from trellis2.renderers.mesh_renderer import MeshRenderer
    from trellis2.representations.mesh import Mesh
    folder = c.codec_folder(uid)
    mesh = trimesh.load(folder / 'target_t2.ply', force='mesh', process=False)
    mesh.apply_transform(np.asarray(c.read(folder / 'transform.json')['t2_to_world']))
    render_mesh = Mesh(torch.tensor(np.array(mesh.vertices), device='cuda', dtype=torch.float32),
                       torch.tensor(np.array(mesh.faces), device='cuda', dtype=torch.int32))
    renderer = MeshRenderer({'resolution': 518, 'near': .01, 'far': 100., 'ssaa': 1, 'antialias': False})
    hits = []
    for frame in group['frames']:
        E = torch.tensor(capture['world_w2c'][frame], device='cuda', dtype=torch.float32)
        K = np.asarray(capture['intrinsics'][frame], dtype=np.float32).copy()
        K[0, 2] += .5; K[1, 2] += .5; K[:2] /= capture['image_side']
        with torch.no_grad(): rendered = renderer.render(render_mesh, E, torch.tensor(K, device='cuda'), return_types=['mask'])
        amodal = rendered['mask'].cpu().numpy() > .5
        with Image.open(c.SOURCE / 'scenes' / group['scene'] / 'masks' / uid / f'{frame:03d}.png') as image:
            visible = np.asarray(image.resize((518, 518), Image.Resampling.NEAREST)) > 127
        hits.append({'frame': frame, 'visible_pixels': int(visible.sum()),
                     'visible_to_amodal_hit': float((visible & amodal).sum() / max(visible.sum(), 1))})
    return hits


def finalize():
    from ...scene_relations.data import GroupDataset
    source = c.read(c.DATA / 'groups.json')
    objects = {r['uid']: r for r in c.read(c.DATA / 'objects.json')['objects']}
    complete = []
    labels = {}; codec_records = []; flagged = []
    for uid in objects:
        folder = c.codec_folder(uid)
        labels[uid] = {key: c.sha(folder / name) for key, name in
                       [('ss_target_file_sha256', 'ss.npz'), ('shape_target_file_sha256', 'shape_512.npz')]}
        assert c.read(folder / 'encode_512.json')['tokens'] <= c.CONFIG['shape_max_tokens']
        ss = c.read(folder / 'ss.json'); codec = c.read(folder / 'codec.json')
        assert ss['finite'] and codec['faces'] > 0
        metrics = codec['metrics']
        issues = []
        if ss['iou'] < .9: issues.append('SS codec IoU below .9')
        if metrics['cd_l1'] > .04: issues.append('codec CD-L1 above .04')
        if metrics['f@0.02'] < .85: issues.append('codec F@.02 below .85')
        codec_records.append({'uid': uid, 'split': objects[uid]['split'], 'ss_iou': ss['iou'],
                              'metrics': metrics, 'issues': issues, 'codec_folder': str(folder)})
        if issues: flagged.append({'uid': uid, 'issues': issues})
    c.write(c.OUT / 'codec_quality.json', {'time': c.now(), 'records': codec_records, 'flagged': flagged,
            'policy': 'Gross label/coordinate checks only, not a model-improvement gate; never silently drop flagged objects'})
    if flagged: raise RuntimeError(f'{len(flagged)} native codec(s) need review; see codec_quality.json. No targets dropped, formal training not launched.')
    for group in source['groups'] + source['pilot_groups']:
        rows = []
        for uid in group['uids']:
            folder = c.codec_folder(uid); cond = c.DATA / 'conditions' / group['id'] / uid
            rows.append({'uid': uid, 'split': group['split'], 'scene': group['scene'],
                         'pose_file': str(cond / 'pose.pt'),
                         'pose_file_sha256': c.read(cond / 'condition.json')['pose_sha256'],
                         'ss_target_file': str(folder / 'ss.npz'), 'shape_target_file': str(folder / 'shape_512.npz'),
                         'gt_mesh': str(folder / 'target_t2.ply'), **labels[uid]})
        complete.append({**group, 'members': rows})
    for split in ('train', 'val', 'test', 'pilot_train', 'pilot_val'):
        is_pilot = split.startswith('pilot_'); raw_split = split.removeprefix('pilot_')
        rows = [g for g in complete if g['split'] == raw_split and g['id'].startswith('pilot_') == is_pilot]
        path = c.DATA / f'{split}.json'
        c.immutable(path, {'plan_identity': c.read(c.OUT / 'plan.json')['identity'], 'groups': rows})
        for stage in ('ss', 'shape'):
            dataset = GroupDataset(path, stage, max_tokens=c.CONFIG['shape_max_tokens'])
            for i in range(len(dataset)): dataset[i]
    c.write(c.OUT / 'data_summary.json', {'time': c.now(), 'anchors': 192, 'objects': len(objects),
            'reused_native': sum(u in c.plan()['native_reuse'] for u in objects),
            'groups': {s: sum(g['split'] == s and not g['id'].startswith('pilot_') for g in complete) for s in ('train', 'val', 'test')},
            'singletons': sum(g.get('singleton', False) for g in complete),
            'gt_pose_controlled': True, 'gt_depth_relation_inputs': False})


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['select', 'assemble', 'reuse', 'codec', 'conditions', 'finalize'])
    p.add_argument('--scene'); p.add_argument('--uid'); p.add_argument('--group'); p.add_argument('--phase'); p.add_argument('--resolution', type=int, default=512)
    a = p.parse_args(); os.environ.update(c.environment())
    if a.action == 'select': select_scene(a.scene)
    elif a.action == 'assemble': assemble()
    elif a.action == 'reuse': validate_reuse(a.uid)
    elif a.action == 'codec': codec(a.phase, a.uid, a.resolution)
    elif a.action == 'conditions': conditions(a.group)
    else: finalize()


if __name__ == '__main__': main()
