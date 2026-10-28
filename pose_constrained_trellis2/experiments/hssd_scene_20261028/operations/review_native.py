"""Post-run review only: no changes to frozen labels, runtime, or checkpoints.

Run with trellis2 Python from Tracker: python -B -m
pose_constrained_trellis2.experiments.hssd_scene_20261028.operations.review_native
All new artifacts go to the project's /data review directory.
"""
import gc
import json
import os
import sys
from pathlib import Path

from .. import common as c


def main():
    os.environ.update(c.environment())
    import numpy as np
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from PIL import Image, ImageDraw, ImageFont
    from trellis2.renderers.mesh_renderer import MeshRenderer
    from trellis2.representations.mesh import Mesh
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from ..runner import identity, stages

    dest = c.OUT / 'review_20261028'
    dest.mkdir(parents=True, exist_ok=True)
    frozen = c.read(c.OUT / 'runtime.json')
    assert c.digest(identity()) == frozen['identity'], 'Frozen runtime changed'
    receipts = []
    for name, _, _, _ in stages():
        receipt = c.read(c.OUT / 'stages' / (name + '.json'))
        assert receipt['runtime_identity'] == frozen['identity']
        for artifact in receipt['artifacts']:
            p = Path(artifact['path'])
            assert p.is_file() and p.stat().st_size == artifact['bytes'], p
        receipts.append(name)
    rows = c.read(c.DATA / 'all.json')['samples']
    hashed = 0
    for row in rows:
        for key in ('pose_file', 'ss_target_file', 'shape_target_file'):
            assert c.sha(row[key]) == row[key + '_sha256'], row[key]
            hashed += 1

    def metrics(a, b):
        da = cKDTree(b).query(a)[0]
        db = cKDTree(a).query(b)[0]
        result = {'cd_l1': float((da.mean() + db.mean()) / 2)}
        for threshold in (.01, .02):
            p, r = (db < threshold).mean(), (da < threshold).mean()
            result[f'f@{threshold}'] = float(2*p*r / max(p+r, 1e-12))
        return result

    renderer = MeshRenderer({'resolution': 384, 'near': .1, 'far': 10., 'ssaa': 1})
    cameras, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        np.deg2rad([25., 145., 265.]).tolist(), np.deg2rad([20., 20., 20.]).tolist(), 2., 40.)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)

    def render(mesh):
        assert np.isfinite(mesh.vertices).all()
        assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
        sample = Mesh(torch.as_tensor(np.array(mesh.vertices), dtype=torch.float32, device='cuda'),
                      torch.as_tensor(np.array(mesh.faces), dtype=torch.int32, device='cuda'))
        frames = []
        with torch.no_grad():
            for camera, K in zip(cameras, intrinsics):
                output = renderer.render(sample, camera, K, return_types=['mask', 'normal'])
                mask = output['mask'].cpu().numpy()[..., None]
                normals = output['normal'].permute(1, 2, 0).cpu().numpy() * 2 - 1
                light = np.array([.25, -.4, .88]); light /= np.linalg.norm(light)
                shade = .3 + .7 * np.abs(normals @ light)[..., None].clip(0, 1)
                rgb = (shade * np.array([.62, .7, .8]) * mask + 1 - mask).clip(0, 1)
                frames.append(Image.fromarray(np.uint8(rgb * 255)))
        del sample, output
        return frames

    def panel(uid, mesh_rows, filename, subtitle):
        # Explicit reuse for this review's interrupted renderer; all numeric
        # and artifact validation below still runs. Never used by training.
        if '--reuse-rendered' in sys.argv and (dest / filename).is_file():
            with Image.open(dest / filename) as saved:
                assert saved.size == (1152, 80 + 410 * len(mesh_rows))
                return saved.copy()
        canvas = Image.new('RGB', (1152, 80 + 410 * len(mesh_rows)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), uid, fill='black', font=font)
        draw.text((12, 35), subtitle, fill='black', font=font)
        for i, (label, mesh) in enumerate(mesh_rows):
            y = 80 + i*410
            draw.text((12, y), label, fill='black', font=font)
            for j, frame in enumerate(render(mesh)):
                canvas.paste(frame, (384*j, y+26))
        canvas.save(dest / filename)
        return canvas

    results, panels = [], []
    for index, source in enumerate(c.plan()['plan']['objects']):
        uid = source['uid']; folder = c.DATA/'objects'/uid
        codec = c.read(folder/'codec.json')
        pose = torch.load(folder/'pose.pt', map_location='cpu', weights_only=False)
        assert tuple(pose['dino_features'].shape) == (len(source['views']), 1029, 1024)
        assert len(source['views']) >= 2
        assert torch.isfinite(pose['dino_features']).all()
        counts = {}
        for resolution in (512, 1024):
            with np.load(folder/f'shape_{resolution}.npz') as label:
                xyz, feats = label['coords'], label['feats']
                assert len(xyz) > 0 and feats.shape == (len(xyz), 32)
                assert np.isfinite(feats).all() and np.issubdtype(xyz.dtype, np.integer)
                assert xyz.min() >= 0 and xyz.max() < resolution//16
                assert len(np.unique(xyz, axis=0)) == len(xyz)
                counts[str(resolution)] = len(xyz)
        with np.load(folder/'ss.npz') as label:
            assert label['z'].shape == (8, 16, 16, 16) and np.isfinite(label['z']).all()
        gt = trimesh.load(folder/'target_t2.ply', force='mesh', process=False)
        decoded = trimesh.load(folder/'decoded_t2.ply', force='mesh', process=False)
        world = trimesh.load(folder/'decoded_world.ply', force='mesh', process=False)
        transform = c.read(folder/'transform.json')
        assert np.array_equal(decoded.faces, world.faces)
        expected = trimesh.transform_points(decoded.vertices, np.array(transform['t2_to_world']))
        world_error = float(np.abs(world.vertices-expected).max())
        assert world_error < 3e-5, (uid, world_error)
        del world, expected
        np.random.seed(20261028)
        # Two independent samples of the SAME GT expose finite-sampling error.
        self_control = metrics(gt.sample(20000), gt.sample(20000))
        figure = f'{uid}_triangles.jpg'
        canvas = panel(uid, [('GT complete mesh', gt), ('Native Shape512 codec', decoded)], figure,
                       'Actual triangles; fixed canonical cameras; no ICP or per-mesh alignment')
        panels.append(canvas.resize((576, 450)))
        result = {'uid': uid, 'split': source['split'], 'category': source['category'], 'views': len(source['views']),
                  'codec_metrics_original': codec['metrics'], 'gt_self_20k_diagnostic': self_control,
                  'ss_iou': codec['ss']['iou'], 'tokens': counts, 'vertices': len(decoded.vertices),
                  'faces': len(decoded.faces), 'normalization_roundtrip_error': transform['roundtrip_max_abs'],
                  'serialized_world_transform_error': world_error, 'figure': str(dest/figure)}
        results.append(result)
        if index == 0:
            generated = trimesh.load(c.OUT/'smoke/generated.ply', force='mesh', process=False)
            np.random.seed(20261028)
            smoke_metrics = metrics(gt.sample(20000), generated.sample(20000))
            panel(uid, [('GT complete mesh', gt), ('Native codec (label roundtrip)', decoded),
                        ('Generated after 3 SS + 3 Shape steps (plumbing check only)', generated)],
                  'smoke_triangles.jpg', 'Same canonical cameras; unaligned raw generation; seed42; 12+12 sampling steps')
            del generated
        print(json.dumps({'review_object': uid, 'completed': index+1, 'total': 12}), flush=True)
        del gt, decoded, pose, canvas
        gc.collect()
    for page in range(2):
        overview = Image.new('RGB', (1152, 1350), 'white')
        for i, card in enumerate(panels[page*6:(page+1)*6]):
            overview.paste(card, ((i % 2)*576, (i//2)*450))
        overview.save(dest/f'codec_overview_{page+1}.jpg')
    del panels, renderer
    torch.cuda.empty_cache()

    smoke = {}
    for stage in ('ss', 'shape'):
        folder = c.OUT/'smoke'/stage/'checkpoints'
        first = torch.load(folder/'step_000001.pt', weights_only=False, map_location='cpu', mmap=True)
        second = torch.load(folder/'step_000002.pt', weights_only=False, map_location='cpu', mmap=True)
        third = torch.load(folder/'step_000003.pt', weights_only=False, map_location='cpu', mmap=True)
        assert third['step'] == 3 and third['history'][:2] == second['history']
        assert [h['step'] for h in third['history']] == [1, 2, 3]
        assert second['data_identity'] == third['data_identity']
        finite = all(bool(torch.isfinite(value).all()) for key in ('model_trainable_state', 'ema_trainable_state')
                     for value in third[key].values())
        assert finite
        changes = {name: float((value-second['model_trainable_state'][name]).abs().max())
                   for name, value in third['model_trainable_state'].items()}
        assert any(v > 0 for v in changes.values())
        smoke[stage] = {'history': third['history'], 'initial_stock_audit': first['model_identity']['initial_stock_audit'],
                        'trainable_surface': first['model_identity']['trainable_surface'], 'finite_raw_and_ema': finite,
                        'resume_history_prefix_identical': True, 'tensors_changed_step2_to3': sum(v > 0 for v in changes.values()),
                        'trainable_tensor_count': len(changes), 'max_parameter_change_step2_to3': max(changes.values())}
        del first, second, third
        gc.collect()
    aggregate = {key: {op: float(fn([r['codec_metrics_original'][key] for r in results]))
                       for op, fn in [('mean', np.mean), ('min', np.min), ('max', np.max)]}
                 for key in ('cd_l1', 'f@0.01', 'f@0.02')}
    self_mean = {key: float(np.mean([r['gt_self_20k_diagnostic'][key] for r in results])) for key in ('cd_l1','f@0.01','f@0.02')}
    report = {'time': c.now(), 'runtime_identity': frozen['identity'], 'verified_stages': len(receipts),
              'verified_manifest_hashes': hashed, 'objects': results, 'codec_aggregate': aggregate,
              'gt_self_20k_diagnostic_mean': self_mean, 'smoke': smoke, 'smoke_generated_metrics': smoke_metrics,
              'sampling': {'points_per_mesh': 20000, 'seed': 20261028, 'alignment': 'none',
                           'interpretation': 'Self-sampling control is diagnostic, not subtracted from codec metrics'},
              'render': {'resolution': 384, 'yaw_degrees': [25,145,265], 'pitch_degrees': [20,20,20],
                         'radius': 2., 'fov_degrees': 40., 'geometry': 'full triangles, no simplification or transform fitting'},
              'scope': 'Codec and smoke execution review; no baseline or relation-model efficacy conclusion'}
    c.write(dest/'review.json', report)
    print(json.dumps({'complete': True, 'report': str(dest/'review.json'), 'codec_aggregate': aggregate,
                      'gt_self_mean': self_mean, 'smoke_generated_metrics': smoke_metrics}), flush=True)


if __name__ == '__main__':
    main()
