"""Post-hoc crop control reusing the unchanged, frozen inference worker."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from visual_evidence_probe import common, data, worker, report
from visual_evidence_probe.common import ARMS, SEEDS, ROOT, read, write, binding, verify, lock_json, now
from visual_evidence_probe.render import render

ORIGINAL = ROOT / 'outputs/correlated_views_20261025'
OUT = ROOT / 'outputs/correlated_views_cropped_control_20261025'
INDICES = (0, 1, 2, 3)


def configure():
    # Process-local dependency injection; preserve every original source/output.
    common.OUT = data.OUT = worker.OUT = report.OUT = OUT
    worker.input_for = input_for
    worker.make_render_features = render_features
    worker.input_gallery = gallery


def folder(index):
    return OUT / 'objects' / f'{index:03d}'


def prepare():
    old = read(ORIGINAL / 'protocol.json')
    sources = old['sources'] + old['dependency_sources'] + [binding(__file__),
        binding(Path(__file__).with_name('CROPPED_PROTOCOL.md')), binding(ROOT / 'run_cropped.sh'),
        binding(ROOT.parent / 'ar_ss_flow/shared_object_preprocessing.py')]
    protocol = {**old, 'format': 'visual_evidence_probe.cropped_control.v1',
        'indices': list(INDICES), 'sources': sources, 'dependency_sources': [],
        'scope': 'post-hoc crop-conformance control on first four inspected objects',
        'parent_protocol': binding(ORIGINAL / 'protocol.json'),
        'cropping': {'resolution': 518, 'margin': 1.1, 'alpha_threshold': .8,
                     'intrinsics': 'affine @ K; synthetic crops use own masks only'},
        'encoder_precision': 'fp32 for all new images; fp16 cache'}
    lock_json(OUT / 'protocol.json', protocol)
    for index in INDICES:
        lock_json(folder(index) / 'object.json', read(ORIGINAL / 'objects' / f'{index:03d}' / 'object.json'))


@torch.no_grad()
def encode_crop(index, key, images, masks, obs, pipeline):
    from ar_ss_flow.shared_object_preprocessing import prepare_shared_object_arrays
    from pose_aligned_layered_reconstruction.paired_encoder import _encode_patch_tokens
    prepared = prepare_shared_object_arrays([np.asarray(im.convert('RGB')) for im in images],
        [np.asarray(m, np.uint8) * 255 if np.asarray(m).dtype == bool else np.asarray(m, np.uint8) for m in masks],
        resolution=518, foreground_margin=1.1, alpha_threshold=.8)
    k = torch.from_numpy(prepared.source_to_feature_affines) @ obs['intrinsics']
    root = folder(index) / 'crop_images' / key
    root.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(prepared.images):
        im.save(root / f'{i:02d}.png')
    # Check the geometry of cropping using synthetic 3D queries, without GT shape.
    rng = np.random.default_rng(20261025)
    points = np.c_[rng.uniform(-.4, .4, (32, 3)), np.ones(32)]
    maximum = 0.
    for j, (old_k, new_k, w) in enumerate(zip(obs['intrinsics'].numpy(), k.numpy(), obs['extrinsics'].numpy())):
        xyz = points @ w.T
        old = xyz[:, :3] @ old_k.T
        new = xyz[:, :3] @ new_k.T
        old = old / old[:, 2:]
        new = new / new[:, 2:]
        expected = old @ prepared.source_to_feature_affines[j].T
        maximum = max(maximum, float(np.abs(new - expected).max()))
    if maximum > 1.e-3:
        raise ValueError(f'Crop camera-affine mismatch: {maximum}')
    chunks = [_encode_patch_tokens(pipeline, prepared.images[i:i+2]).half().cpu() for i in range(0, 8, 2)]
    write(root / 'audit.json', {**prepared.geometry_record(), 'max_projection_error_pixels': maximum,
                               'dino_precision': 'fp32', 'mask_source': 'own input/render only'})
    return {'features': torch.cat(chunks), 'intrinsics': k.float()}


@torch.no_grad()
def original_features(index):
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder
    path = folder(index) / 'original_crop.pt'
    if path.exists():
        verify(read(path.with_suffix('.json'))['features'])
        return
    rec, obs = data.object_data(index)
    images = [Image.open(verify(b)).convert('RGB') for b in rec['images']]
    masks = [np.asarray(Image.open(verify(b)).convert('L')) for b in rec['masks']]
    pipeline = build_frozen_dino_encoder(device=torch.device('cuda:0'))
    cached = encode_crop(index, 'original', images, masks, obs, pipeline)
    torch.save(cached, path)
    write(path.with_suffix('.json'), {'features': binding(path)})
    del pipeline
    gc.collect(); torch.cuda.empty_cache()


def input_for(index, arm):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    rec, obs = data.object_data(index)
    original = torch.load(folder(index) / 'original_crop.pt', map_location='cpu', weights_only=False)
    inds = [0, 1] if arm == 'real2' else ([0, 1] * 4 if arm == 'repeat8' else list(range(8)))
    if arm in ('real2', 'repeat8', 'real8'):
        features, k = original['features'][inds].float(), original['intrinsics'][inds]
    else:
        cache = torch.load(folder(index) / 'render_features.pt', map_location='cpu', weights_only=False)
        if arm in ('gt_render6', 'donor_a6', 'donor_b6'):
            key = {'gt_render6': 'gt', 'donor_a6': 'a', 'donor_b6': 'b'}[arm]
            extra, extra_k = cache[key]['features'][2:], cache[key]['intrinsics'][2:]
        else:
            offset = 0 if arm == 'mixed_ab6' else 1
            keys = ['a' if (j + offset) % 2 == 0 else 'b' for j in range(6)]
            extra = torch.stack([cache[key]['features'][j+2] for j, key in enumerate(keys)])
            extra_k = torch.stack([cache[key]['intrinsics'][j+2] for j, key in enumerate(keys)])
        features = torch.cat((original['features'][:2], extra)).float()
        k = torch.cat((original['intrinsics'][:2], extra_k))
    result = {key: obs[key] for key in data.OBS_KEYS}
    result.update(intrinsics=k, extrinsics=obs['extrinsics'][inds])
    result.update(build_masked_dino_contexts(features))
    result['visual_patch_features'] = features
    return result


@torch.no_grad()
def render_features(index, obs, gt, context, device):
    import trimesh
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder
    path = folder(index) / 'render_features.pt'
    if path.exists():
        verify(read(path.with_suffix('.json'))['features'])
        return
    pipeline = build_frozen_dino_encoder(device=device)
    cache = {}
    meshes = [('gt', gt)] + [(key, trimesh.load(folder(index) / f'results/real2/{seed}.obj', force='mesh', process=False))
                           for key, seed in [('a', 42), ('b', 43)]]
    for key, mesh in meshes:
        images, masks = render(mesh, obs['intrinsics'], obs['extrinsics'], side=518, context=context)
        raw = folder(index) / 'renders_raw' / key
        raw.mkdir(parents=True, exist_ok=True)
        for j, im in enumerate(images):
            im.save(raw / f'{j:02d}.png')
        np.savez_compressed(raw / 'masks.npz', masks=masks)
        cache[key] = encode_crop(index, key, images, masks, obs, pipeline)
    torch.save(cache, path)
    write(path.with_suffix('.json'), {'features': binding(path), 'encoding': 'FP32 DINO; standard shared bbox crop'})
    del pipeline
    gc.collect(); torch.cuda.empty_cache()


def gallery(index):
    size, label, header = 160, 150, 24
    canvas = Image.new('RGB', (label + size*8, (size+header)*8), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, arm in enumerate(ARMS):
        draw.text((5, row*(size+header)+60), arm, fill='black')
        for pos in range(2 if arm == 'real2' else 8):
            if arm in ('real2', 'repeat8', 'real8') or pos < 2:
                key, i = 'original', pos % 2 if arm == 'repeat8' else pos
            else:
                key = {'gt_render6': 'gt', 'donor_a6': 'a', 'donor_b6': 'b'}.get(arm)
                if key is None:
                    key = 'a' if (pos - 2 + (arm == 'mixed_ba6')) % 2 == 0 else 'b'
                i = pos
            im = Image.open(folder(index) / 'crop_images' / key / f'{i:02d}.png').convert('RGB')
            x, y = label + pos*size, row*(size+header)
            canvas.paste(ImageOps.contain(im, (size,size)), (x,y+header))
            draw.text((x+3,y+3), str(pos), fill='black')
    canvas.save(folder(index) / 'inputs.jpg', quality=90)


def summarize():
    from visual_evidence_probe.report import descriptive_delta
    rows = {(i, a, s): read(folder(i) / f'results/{a}/{s}.json') for i in INDICES for a in ARMS for s in SEEDS}
    for row in rows.values():
        for key in ('mesh', 'coords', 'gt'):
            verify(row[key])
    getters = {'cd': lambda r: r['surface']['chamfer_l1'], 'f02': lambda r: r['surface']['fscore_0p02'],
               'ss_iou': lambda r: r['occupancy']['iou'], 'silhouette_source2': lambda r: r['silhouette']['source2']}
    obj = {a: {k: [float(np.mean([g(rows[i,a,s]) for s in SEEDS])) for i in INDICES] for k,g in getters.items()} for a in ARMS}
    means = {a: {k: float(np.mean(v)) for k,v in values.items()} for a,values in obj.items()}
    balanced = {k: descriptive_delta((np.array(obj['donor_a6'][k])+obj['donor_b6'][k]-obj['mixed_ab6'][k]-obj['mixed_ba6'][k])*.5) for k in getters}
    result = {'time': now(), 'objects': len(INDICES), 'records': len(rows), 'means': means, 'by_object': obj,
              'balanced_coherent_minus_mixed': balanced, 'scope': 'post-hoc training-crop control, first four objects',
              'deltas': {a: {k: descriptive_delta(np.array(obj[a][k])-obj['real2'][k]) for k in getters} for a in ARMS if a != 'real2'}}
    write(OUT / 'summary.json', result)
    lines = ['# Training-crop control', '', result['time'], '', result['scope'], '',
             '|Arm|CD↓|F@.02↑|SS IoU↑|source2 silhouette↑|', '|---|---:|---:|---:|---:|']
    for a,m in means.items():
        lines.append(f"|{a}|{m['cd']:.6f}|{m['f02']:.4f}|{m['ss_iou']:.4f}|{m['silhouette_source2']:.4f}|")
    lines += ['', 'Balanced coherent-minus-mixed CD: '+str(balanced['cd']), '',
              'Four previously inspected objects; post-hoc preprocessing control, no heldout/causal-confidence claim.',
              'All synthetic crops use their own render masks. No hidden GT crop, fitting, alignment or ICP.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


def run(gpus):
    import fcntl
    prepare()
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = []
        for i,gpu in zip(INDICES,gpus):
            log = OUT / f'{i:03d}_{time.time_ns()}.log'
            stream = log.open('w')
            process = subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'worker','--index',str(i)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)), stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            active.append((i,gpu,process,stream))
        while any(p.poll() is None for _,_,p,_ in active):
            write(OUT / 'status.json', {'time':now(),'status':'running',
                'active':[{'index':i,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for i,g,p,_ in active]})
            time.sleep(3)
        failed = []
        for i,g,p,stream in active:
            stream.close()
            if p.returncode: failed.append({'index':i,'gpu':g,'exit_code':p.returncode})
        if failed:
            write(OUT / 'status.json', {'time':now(),'status':'failed','failed':failed})
            raise RuntimeError(str(failed))
        summarize()
        write(OUT / 'status.json', {'time':now(),'status':'complete_review_required'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run','worker','status','report'))
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--gpus',default='0,1,4,5')
    args = parser.parse_args()
    configure()
    if args.action == 'worker':
        from visual_evidence_probe.runtime import setup
        setup(); original_features(args.index); worker.run(args.index)
    elif args.action == 'run': run([int(g) for g in args.gpus.split(',')])
    elif args.action == 'report': summarize()
    else:
        print(read(OUT / 'status.json') if (OUT / 'status.json').exists() else {'status':'not_started'})
        print('mesh_records',len(list((OUT/'objects').glob('*/results/*/*.json'))),'/64')


if __name__ == '__main__':
    main()
