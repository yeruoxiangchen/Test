"""Paired object-level summaries; keep limitations next to the conclusions."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import OUT, ARMS, SEEDS, now, read, write, verify, folder


def descriptive_delta(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(20261025)
    boot = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'negative_objects': int((values < -1.e-9).sum()), 'positive_objects': int((values > 1.e-9).sum()),
            'within_scene_object_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'per_object': values.tolist()}


def mesh_gallery(index):
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from .data import object_data
    from .render import render
    rec, obs = object_data(index)
    context = dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    rows = [('GT', rec['gt']['path'])]
    rows += [(f'{arm} / seed{seed}', read(folder(index) / 'results' / arm / f'{seed}.json')['mesh']['path'])
             for arm in ARMS for seed in SEEDS]
    h, w, label = 220, 220, 210
    sheet = Image.new('RGB', (label + 4 * w, len(rows) * h), 'white')
    draw = ImageDraw.Draw(sheet)
    views = [0, 1, 3, 6]
    for i, (name, path) in enumerate(rows):
        mesh = trimesh.load(path, force='mesh', process=False)
        images, _ = render(mesh, obs['intrinsics'][views], obs['extrinsics'][views], side=220, context=context)
        draw.text((8, i * h + 60), name, fill='black')
        if i:
            arm, seed_text = name.split(' / seed')
            score = read(folder(index) / 'results' / arm / f'{seed_text}.json')['surface']
            draw.text((8, i * h + 90), f"CD {score['chamfer_l1']:.5f}", fill='black')
            draw.text((8, i * h + 110), f"F@.02 {score['fscore_0p02']:.3f}", fill='black')
        for j, im in enumerate(images):
            sheet.paste(im, (label + j * w, i * h))
    path = folder(index) / 'mesh_comparison.jpg'
    sheet.save(path, quality=92)
    return path


def summarize():
    import trimesh
    from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics
    records = {}
    for index in range(8):
        for arm in ARMS:
            for seed in SEEDS:
                path = folder(index) / 'results' / arm / f'{seed}.json'
                if not path.exists():
                    raise ValueError(f'Incomplete diagnostic: {path}')
                r = read(path)
                if (r['index'], r['arm'], r['seed']) != (index, arm, seed) or not r['success']:
                    raise ValueError('Invalid/mismatched result')
                for key in ('mesh', 'coords', 'gt'):
                    verify(r[key])
                records[index, arm, seed] = r

    getters = {'cd': lambda r: r['surface']['chamfer_l1'],
               'f02': lambda r: r['surface']['fscore_0p02'],
               'ss_iou': lambda r: r['occupancy']['iou'],
               'silhouette_source2': lambda r: r['silhouette']['source2'],
               'silhouette_other6': lambda r: r['silhouette']['remaining6']}
    by_object = {arm: {key: [float(np.mean([get(records[i, arm, s]) for s in SEEDS])) for i in range(8)]
                       for key, get in getters.items()} for arm in ARMS}
    means = {arm: {key: float(np.mean(vals)) for key, vals in v.items()} for arm, v in by_object.items()}
    deltas = {arm + '_minus_real2': {key: descriptive_delta(np.asarray(by_object[arm][key]) - by_object['real2'][key])
                                  for key in getters} for arm in ARMS if arm != 'real2'}
    correlation = {}
    for key in getters:
        coherent = .5 * (np.array(by_object['donor_a6'][key]) + by_object['donor_b6'][key])
        mixed = .5 * (np.array(by_object['mixed_ab6'][key]) + by_object['mixed_ba6'][key])
        correlation[key] = descriptive_delta(coherent - mixed)

    agreement, attraction = {}, {}
    for index in range(8):
        meshes = {(a, s): trimesh.load(records[index, a, s]['mesh']['path'], force='mesh', process=False)
                  for a in ARMS for s in SEEDS}
        for arm in ARMS:
            value = surface_metrics(meshes[arm, 42], meshes[arm, 43], count=20000, seed=20261025)['chamfer_l1']
            agreement[index, arm] = value
            if arm in ('donor_a6', 'donor_b6'):
                donor_seed = 42 if arm == 'donor_a6' else 43
                other_seed = 43 if donor_seed == 42 else 42
                # Compare the OTHER output seed to the donor, avoiding the trivial self-distance.
                before = surface_metrics(meshes['real2', other_seed], meshes['real2', donor_seed], count=20000, seed=20261025)['chamfer_l1']
                after = surface_metrics(meshes[arm, other_seed], meshes['real2', donor_seed], count=20000, seed=20261025)['chamfer_l1']
                attraction[index, arm] = {'before_cd_to_donor': before, 'after_cd_to_donor': after,
                                         'delta': after - before}

    result = {'time': now(), 'scope': '8 pre-inspected objects from one HSSD scene; exploratory, no heldout claim',
        'records': len(records), 'objects': 8, 'seeds': list(SEEDS), 'means': means, 'by_object': by_object,
        'deltas': deltas, 'balanced_coherent_minus_mixed': correlation,
        'seed_agreement_cd': {arm: {'mean': float(np.mean([agreement[i, arm] for i in range(8)])),
                                  'per_object': [agreement[i, arm] for i in range(8)]} for arm in ARMS},
        'donor_attraction': {f'{i:03d}_{arm}': value for (i, arm), value in attraction.items()},
        'confidence_claim_tested': False, 'alignment': 'none', 'surface_samples': 50000,
        'failure_filtering': False}
    write(OUT / 'summary.json', result)
    lines = ['# Correlated synthetic-view probe', '', f'Completed: {result["time"]}', '',
        '8 previously inspected HSSD objects in one scene; frozen SS30K/SLat30K EMA. Two paired sampling seeds.',
        'Original canonical GT; no fitted alignment. This is a hypothesis diagnostic, not independent generalization.', '',
        '| Arm | CD ↓ | F@.02 ↑ | SS IoU ↑ | source2 silhouette ↑ | other6 silhouette ↑ | seed-pair CD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = means[arm]
        lines.append(f"| {arm} | {m['cd']:.6f} | {m['f02']:.4f} | {m['ss_iou']:.4f} | {m['silhouette_source2']:.4f} | {m['silhouette_other6']:.4f} | {result['seed_agreement_cd'][arm]['mean']:.6f} |")
    lines += ['', '## Balanced source-correlation comparison', '',
        'Compare average(donor_a6, donor_b6) against average(mixed_ab6, mixed_ba6).',
        'The donor-image marginals and six target cameras match across these paired averages.',
        'Positive CD difference means coherent donor groups are worse; negative means better.',
        'Different seeds are not proven independent sources. Nonlinear input mixing is part of this intervention.', '']
    for key, value in correlation.items():
        lines.append(f"- {key}: coherent minus mixed = {value['mean']:+.6f}; exploratory within-scene object bootstrap {value['within_scene_object_bootstrap_ci95']}")
    lines += ['', '## Interpretation boundaries', '',
        '- Synthetic-input harm alone does not establish correlated-error amplification.',
        '- gt_render6 is a privileged rendering-domain control, not a usable algorithm.',
        '- repeat8 changes SS context token subsampling; inspect this before interpreting a difference.',
        '- Seed-pair agreement is not calibrated geometric confidence; no confidence claim is tested.',
        '- Surface points and SS support represent different resolutions; do not substitute one metric for the other.',
        '- Four displayed cameras are shared for every mesh. Images are for review, not selected successful cases.', '',
        '## Artifacts', '', '[Full numerical summary](summary.json)', '']
    for index in range(8):
        lines += [f'- Object {index:03d}: [inputs](objects/{index:03d}/inputs.jpg), [mesh comparison](objects/{index:03d}/mesh_comparison.jpg), [OBJ files](objects/{index:03d}/results).']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {OUT / "REPORT.md"}', flush=True)
