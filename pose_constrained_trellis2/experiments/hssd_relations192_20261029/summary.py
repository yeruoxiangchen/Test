"""Anchor-level paired metrics, triangle meshes, and joint visible-owner checks."""
import argparse
import os
from statistics import mean
from . import common as c


def evaluation_groups(test=False):
    if test: return c.read(c.DATA / 'test.json')['groups']
    training = sorted(c.read(c.DATA / 'train.json')['groups'], key=lambda g: c.digest(['train_diagnostic', g['id']]))[:8]
    return training + c.read(c.DATA / 'val.json')['groups']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--test', action='store_true'); a = p.parse_args()
    os.environ.update(c.environment())
    import numpy as np
    import torch, trimesh
    from PIL import Image, ImageDraw, ImageFont
    from trellis2.renderers.mesh_renderer import MeshRenderer
    from trellis2.representations.mesh import Mesh
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    canonical = MeshRenderer({'resolution': 384, 'near': .1, 'far': 10., 'ssaa': 1})
    scene_renderer = MeshRenderer({'resolution': 518, 'near': .01, 'far': 100., 'ssaa': 1, 'antialias': False})
    cameras, Ks = yaw_pitch_r_fov_to_extrinsics_intrinsics(np.deg2rad([25., 145., 265.]).tolist(),
                                                        np.deg2rad([20., 20., 20.]).tolist(), 2., 40.)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    figures = c.OUT / ('test_figures' if a.test else 'figures'); figures.mkdir(parents=True, exist_ok=True)
    rows = []; scene_rows = []; runtime_identity = c.identity()['identity']
    def mesh_gpu(path):
        m = trimesh.load(path, force='mesh', process=False)
        return Mesh(torch.tensor(np.array(m.vertices), device='cuda', dtype=torch.float32),
                    torch.tensor(np.array(m.faces), device='cuda', dtype=torch.int32))
    def shaded(mesh, E, K):
        with torch.no_grad(): rendered = canonical.render(mesh, E, K, return_types=['mask', 'normal'])
        mask = rendered['mask'].cpu().numpy()[..., None]
        normals = rendered['normal'].permute(1, 2, 0).cpu().numpy() * 2 - 1
        light = np.asarray([.25, -.4, .88]); light /= np.linalg.norm(light)
        shade = .3 + .7 * np.abs(normals @ light)[..., None].clip(0, 1)
        return Image.fromarray(np.uint8((shade * np.array([.62, .7, .8]) * mask + 1 - mask).clip(0, 1) * 255))
    for group in evaluation_groups(a.test):
        for seed in c.CONFIG['evaluation_seeds']:
            cases = {arm: c.read(c.OUT / 'evaluations' / arm / group['id'] / f'seed_{seed}' / 'result.json')
                     for arm in c.CONFIG['arms']}
            for arm, case in cases.items():
                assert case['runtime_identity'] == runtime_identity and not case['smoke']
                record = next(r for r in case['objects'] if r['primary_anchor'])
                rows.append({'uid': group['anchor'], 'split': group['split'], 'scene': group['scene'],
                             'arm': arm, 'seed': seed, **record})
            canvas = Image.new('RGB', (1152, 1290), 'white'); draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), f'{group["anchor"]} | {group["split"]} | seed {seed} | no fitted alignment', fill='black', font=font)
            gt = next(r['gt_mesh'] for r in group['members'] if r['uid'] == group['anchor'])
            paths = [('GT', gt)] + [(arm, next(r['mesh'] for r in case['objects'] if r['primary_anchor'])) for arm, case in cases.items()]
            for k, (label, path) in enumerate(paths):
                y = 48 + k * 414; draw.text((10, y), label if path else label + ' FAILED', fill='black', font=font)
                if path:
                    mesh = mesh_gpu(path)
                    for j, (E, K) in enumerate(zip(cameras, Ks)): canvas.paste(shaded(mesh, E, K), (384 * j, y + 28))
                    del mesh; torch.cuda.empty_cache()
            canvas.save(figures / f'{group["id"]}_seed{seed}.jpg')
            # Score ownership only where one of the selected instances is observed.
            # Pixels outside this union may contain unmodeled occluders and are unknown.
            capture = c.read(c.SOURCE / 'scenes' / group['scene'] / 'plan.json')['plan']
            world = {arm: {r['uid']: mesh_gpu(r['world_mesh']) for r in case['objects'] if r['success']}
                     for arm, case in cases.items()}
            colors = np.array([[255, 255, 255], [235, 75, 55], [45, 145, 240]], dtype=np.uint8)
            panel = Image.new('RGB', (1554, 558 * min(2, len(group['frames']))), 'white'); pen = ImageDraw.Draw(panel)
            accuracy = {arm: [] for arm in c.CONFIG['arms']}
            for frame_index, frame in enumerate(group['frames']):
                E = torch.tensor(capture['world_w2c'][frame], device='cuda', dtype=torch.float32)
                K = np.asarray(capture['intrinsics'][frame], dtype=np.float32).copy()
                K[0, 2] += .5; K[1, 2] += .5; K[:2] /= capture['image_side']
                K = torch.tensor(K, device='cuda')
                observed = np.zeros((518, 518), np.int32)
                for i, uid in enumerate(group['uids']):
                    with Image.open(c.SOURCE / 'scenes' / group['scene'] / 'masks' / uid / f'{frame:03d}.png') as image:
                        mask = np.asarray(image.resize((518, 518), Image.Resampling.NEAREST)) > 127
                    assert not ((observed > 0) & mask).any(), 'Visible instance masks overlap'
                    observed[mask] = i + 1
                predictions = []
                for arm in c.CONFIG['arms']:
                    depths = []
                    for uid in group['uids']:
                        if uid not in world[arm]: depths.append(np.full((518, 518), np.inf)); continue
                        with torch.no_grad(): result = scene_renderer.render(world[arm][uid], E, K, return_types=['mask', 'depth'])
                        m = result['mask'].cpu().numpy() > .5; d = result['depth'].cpu().numpy()
                        depths.append(np.where(m, d, np.inf))
                    stack = np.stack(depths); owner = np.argmin(stack, axis=0) + 1
                    owner[~np.isfinite(stack.min(axis=0))] = 0
                    valid = observed > 0
                    if not valid.any(): raise RuntimeError('Selected group frame has no visible input mask')
                    accuracy[arm].append(float((owner[valid] == observed[valid]).mean()))
                    predictions.append(owner)
                if frame_index < 2:
                    for col, (label, owner) in enumerate(zip(['observed instance masks'] + c.CONFIG['arms'], [observed] + predictions)):
                        y = frame_index * 558
                        pen.text((col * 518 + 5, y + 5), f'{label}; frame {frame}', fill='black', font=font)
                        panel.paste(Image.fromarray(colors[owner]), (col * 518, y + 30))
            panel.save(figures / f'{group["id"]}_seed{seed}_joint.jpg')
            for arm in c.CONFIG['arms']:
                scene_rows.append({'uid': group['anchor'], 'split': group['split'], 'scene': group['scene'],
                                   'arm': arm, 'seed': seed, 'visible_owner_accuracy': mean(accuracy[arm])})
            del world; torch.cuda.empty_cache()
            print(f'REVIEW_FIGURE {group["id"]} seed={seed}', flush=True)
    aggregates = {}
    for split in sorted({r['split'] for r in rows}):
        aggregates[split] = {}
        for arm in c.CONFIG['arms']:
            selected = [r for r in rows if r['split'] == split and r['arm'] == arm]
            aggregates[split][arm] = {'objects': len({r['uid'] for r in selected}), 'cases': len(selected),
                'successes': sum(r['success'] for r in selected),
                **{k: mean(r['metrics'][k] for r in selected) for k in ('cd_l1', 'f@0.01', 'f@0.02', 'world_cd_l1')},
                'visible_owner_accuracy': mean(r['visible_owner_accuracy'] for r in scene_rows if r['split'] == split and r['arm'] == arm)}
    name = 'test_summary' if a.test else 'summary'
    c.write(c.OUT / f'{name}.json', {'time': c.now(), 'aggregates': aggregates, 'cases': rows,
            'joint_visibility': scene_rows, 'runtime_identity': runtime_identity,
            'scope': 'Primary anchors only; average sampling seeds per anchor; report scenes separately for statistical inference',
            'visible_owner_metric': 'Mean frame accuracy restricted to observed selected-instance pixels; not amodal IoU or scene completeness',
            'gt_pose_controlled': True, 'fitted_alignment': False})
    lines = ['# HSSD TRELLIS.2 group reconstruction', '',
             'Fixed raw weights; same data/groups/exposures for independent and relational arms. GT camera/object transforms.', '',
             '| split | arm | success/cases | CD-L1 | F@.02 | observed owner accuracy |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for split, arms in aggregates.items():
        for arm, r in arms.items(): lines.append(f'| {split} | {arm} | {r["successes"]}/{r["cases"]} | {r["cd_l1"]:.6f} | {r["f@0.02"]:.6f} | {r["visible_owner_accuracy"]:.6f} |')
    lines += ['', 'Train uses eight fixed diagnostic anchors; val/test use every frozen anchor. No per-object alignment.',
              'Neighbors are generated and visualized but not counted again as independent primary test cases.',
              'Joint ownership scores only observed selected-instance pixels, not unknown background or fully hidden surfaces.',
              'Nonempty meshes are execution success, not a quality acceptance gate.', '']
    (c.OUT / ('TEST_REPORT.md' if a.test else 'REPORT.md')).write_text('\n'.join(lines))


if __name__ == '__main__': main()
