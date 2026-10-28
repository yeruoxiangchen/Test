from pathlib import Path

from visual_evidence_probe.common import ROOT, binding, read, write, verify, lock_json, now

OUT = ROOT / 'outputs/visible_error_localization_20261026'
OMNI = Path('/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_upright_random4_20260824_v2')
PRIOR = ROOT / 'outputs/input_consistency_baselines_20261025'
SEEDS = (42, 43)
PARAMS = dict(side=128, feature_side=37, depth_thresholds=[.01, .02, .05],
              primary_threshold=.02, self_visibility_tolerance=.01, erosion_pixels=2,
              boundary_decay_pixels=3., top_fraction=.20, shuffle_seed=20261026,
              surface_samples=50000, defect_sigma=.12, defect_displacement=.06)


def folder(index):
    return OUT / 'objects' / f'{index:03d}'


def emit(index, stage, **kw):
    r = dict(time=now(), index=index, stage=stage, **kw)
    write(OUT / 'workers' / f'{index:03d}.json', r)
    print(r, flush=True)


def prepare():
    import numpy as np
    from PIL import Image
    from visual_evidence_probe.experiments.input_consistency.config import verify_protocol
    verify_protocol()
    manifest = read(OMNI / 'manifest.json')
    categories = sorted({r['category'] for r in manifest['objects']})[:8]
    selected = [min((r for r in manifest['objects'] if r['category'] == c), key=lambda r: r['uid'])
                for c in categories]
    sources = [binding(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    sources += [binding(Path(__file__).with_name('PROTOCOL.md')), binding(ROOT / 'run_localization.sh')]
    prior = read(PRIOR / 'protocol.json')
    protocol = dict(format='visible_error_localization.v1', params=PARAMS,
        sources=sources, inherited=prior['sources']+prior['inherited_sources'],
        prior=binding(PRIOR / 'protocol.json'), source_manifest=binding(OMNI / 'manifest.json'),
        selected_uids=[r['uid'] for r in selected], indices=list(range(12)), seeds=list(SEEDS),
        selection='first UID of each first eight lexicographic categories, no metrics inspected',
        scope='8 new-to-this-probe Omni scanned objects with rendered inputs; 4 old HSSD development objects separate',
        weights='frozen original SS30K/SLat30K EMA; no LaRI and no training',
        labels='evaluation-only visible camera-z error after input-mask global similarity fitting',
        no_hidden_surface_claim=True, ranking_selection_uses_gt=False,
        controls=['patch-position shuffled reference DINO', 'correct GT geometry', 'known local GT-geometry defect'])
    lock_json(OUT / 'protocol.json', protocol)
    for index in range(4):
        r = read(PRIOR / 'objects' / f'{index:03d}' / 'observations.json')
        base = read(ROOT / 'outputs/correlated_views_cropped_control_20261025/objects' / f'{index:03d}' / 'object.json')
        rec = dict(index=index, uid=r['uid'], split='hssd_development', category='scene_object',
            intrinsics=r['intrinsics'][:2], extrinsics=r['extrinsics'][:2], masks=r['masks'],
            images=base['images'][:2], existing_fitted={str(s):binding(PRIOR / 'objects' / f'{index:03d}' / f'refined/{k}.obj')
              for s,k in [(42,'a'),(43,'b')]})
        lock_json(folder(index) / 'observations.json', rec)
        lock_json(folder(index) / 'evaluation_target.json', dict(mesh=base['gt'], transform=np.eye(4).tolist()))
    for index, row in enumerate(selected, 4):
        report = read(verify({'path':row['object_report'],'sha256':row['object_report_sha256']}))
        files = report['rendered_files']
        if [f['view_index'] for f in files] != row['selected_input_view_indices']:
            raise ValueError('camera/image order mismatch')
        root = folder(index) / 'raw'
        root.mkdir(parents=True, exist_ok=True)
        images, masks = [], []
        for j, f in enumerate(files):
            image = Image.open(verify(f['rgba'])).convert('RGBA')
            mask = np.asarray(Image.open(row['mask_images'][j]).convert('L'))
            image = Image.fromarray(np.asarray(image)[:,:,:3]).resize((518,518), Image.Resampling.BICUBIC)
            m = Image.fromarray(mask).resize((518,518), Image.Resampling.NEAREST)
            image.save(root / f'{j:02d}.png'); m.save(root / f'{j:02d}_mask.png')
            images.append(binding(root / f'{j:02d}.png')); masks.append(binding(root / f'{j:02d}_mask.png'))
        k = np.array(row['intrinsic'], np.float32)
        scale = 518 / 512
        resize = np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1]], np.float32)
        k = resize @ k
        rec = dict(index=index, uid=row['uid'], split='omni_test', category=row['category'],
            intrinsics=np.repeat(k[None],4,0).tolist(),
            extrinsics=[np.linalg.inv(np.array(f['camera']['c2w_opencv_model_o'])).tolist() for f in files],
            masks=masks, images=images, source_report=binding(row['object_report']),
            resize='512 to518 pixel-center affine; same geometry for mask and RGB')
        lock_json(folder(index) / 'observations.json', rec)
        lock_json(folder(index) / 'evaluation_target.json', dict(mesh=binding(row['source_mesh']),
                  transform=report['normalization']['source_to_model_o_4x4']))
    write(OUT / 'status.json',dict(time=now(),stage='prepared',new_objects=8,new_meshes=16,development_meshes=8))


def check():
    p = read(OUT / 'protocol.json')
    for b in p['sources']+p['inherited']+[p['prior'],p['source_manifest']]: verify(b)
    if p['params'] != PARAMS:
        raise ValueError('changed protocol')


def observation(index):
    import numpy as np
    import torch
    from PIL import Image
    r = read(folder(index) / 'observations.json')
    rgb = [Image.open(verify(b)).convert('RGB') for b in r['images']]
    masks = np.stack([np.asarray(Image.open(verify(b)).convert('L'))>127 for b in r['masks']])
    return r, torch.tensor(r['intrinsics'],dtype=torch.float32), torch.tensor(r['extrinsics'],dtype=torch.float32),rgb,masks
