"""Bounded, resumable SS/mesh diagnostics with paired noise and explicit cohorts."""
import os
for key, value in {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'SPCONV_ALGO': 'native',
                   'CUDA_HOME': '/usr/local/cuda-12.6', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}.items():
    os.environ.setdefault(key, value)

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...paired_dataset import sha256_file, canonical_json_sha256
from ...prepare_official_experiment import atomic_json
from ...v2.dataset import SceneDataset, select_views

PACKAGE = Path(__file__).resolve().parents[2]
OLD = PACKAGE / 'outputs/round1_20260909'
SOURCE = PACKAGE / 'outputs/round2_v2_overfit32_20261017'
ANALYSIS = SOURCE / 'analysis_20261018'
OUT = PACKAGE / 'outputs/round2_v2_diagnostics_20261018'
ARMS = ('raw_joint', 'raw_frozen', 'raw_frozen_v1recipe', 'raw_frozen_v1recipe_tol004')
KEYS = ('format', 'uid', 'object_uid', 'raw_visual_patch_features', 'masked_visual_patch_features',
        'visible_target_mask', 'intrinsics', 'extrinsics', 'image_size', 'extrinsics_type',
        'camera_forward_sign', 'layer_camera_y_sign', 'grid_transform')


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def verify(item):
    if sha256_file(item['path']) != item['sha256']:
        raise ValueError('changed input: ' + item['path'])
    return item['path']


def bind_code():
    files = list((PACKAGE / 'v2').glob('*.py'))
    files += [Path(__file__), PACKAGE/'frozen_slat.py', PACKAGE/'native_ss.py', PACKAGE/'mesh_metrics.py']
    return {str(p.relative_to(PACKAGE)): sha256_file(p) for p in files}


def original_checkpoint(mode):
    if mode == 'baseline500':
        return SOURCE / 'ss/checkpoints/step_000500.pt'
    if mode in ARMS:
        return OUT / mode / 'checkpoints/step_000500.pt'
    return SOURCE / 'ss/checkpoints/step_002000.pt'


def cohort_indices(kind, length):
    if kind == 'paired32':
        if length != 96:
            raise ValueError('paired32 cohort requires the original ordered 96 samples')
        return list(range(32)) + list(range(32, 96, 2))
    return list(range(length))


def prepare_mesh():
    path = OUT / 'mesh/plan.json'
    if path.exists():
        raise FileExistsError(path)
    manifest = SceneDataset(SOURCE/'train96.json', require_labels=False)
    rows = []
    for i in cohort_indices('paired32', len(manifest)):
        r = manifest.rows[i]
        original = manifest[i]
        cases = []
        for seed in (42, 43):
            for mode in ('parent', 'v1', 'raw'):
                if mode == 'v1' and i < 32:
                    paths = [OLD / f'overfit32_candidate2000/coordinates/{i:05d}_{seed}_candidate.npz']
                else:
                    folder = 'v1_coordinates' if mode == 'v1' else 'ss_coordinates'
                    paths = list((ANALYSIS/folder).glob(f'*/{i:03d}_{seed}_{mode}.npz'))
                if len(paths) != 1:
                    raise ValueError(f'missing/ambiguous SS coordinates: {i} {seed} {mode}')
                cases.append({'seed': seed, 'mode': mode, 'coordinates': binding(paths[0])})
        rows.append({'index': i, 'object_index': i if i < 32 else (i-32)//2,
                     'uid': original['uid'], 'object_uid': original['object_uid'],
                     'domain': original['data_domain'], 'sample': binding(r['sample']),
                     'official_slat': binding(original['source_binding']['slat']), 'cases': cases})
    plan = {'manifest': binding(SOURCE/'train96.json'), 'rows': rows,
            'cohort': 'all32 objects: clean and first occlusion variant, selected before mesh results',
            'modes': ['parent', 'v1', 'raw'], 'seeds': [42, 43], 'surface_samples': 20000,
            'thresholds': [.01, .02, .05], 'code': bind_code(),
            'gt': 'official SLat VAE proxy in native frame; no alignment or rescaling',
            'expected_records': len(rows)*6, 'slat_steps': 25,
            'ss_comparison': binding(ANALYSIS/'v1_v2_comparison.json')}
    atomic_json(path, {'plan': plan, 'identity_hash': canonical_json_sha256(plan)})
    print(json.dumps({'mesh_plan': str(path), 'expected_records': plan['expected_records']}), flush=True)


@torch.no_grad()
def mesh(args):
    from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent
    from ...evaluate_mesh import decode_with_flow_offload
    from ...mesh_metrics import surface_metrics, structure_metrics
    import trimesh
    root = OUT / 'mesh'
    payload = json.loads((root/'plan.json').read_text()); plan = payload['plan']
    identity = payload['identity_hash']
    if canonical_json_sha256(plan) != identity:
        raise ValueError('mesh plan hash mismatch')
    for name, value in plan['code'].items():
        if sha256_file(PACKAGE/name) != value:
            raise ValueError('code changed after preparing mesh: ' + name)
    dataset = SceneDataset(verify(plan['manifest']), require_labels=False)
    device = torch.device('cuda:0')
    # These are training objects: record overlap explicitly instead of claiming held-out evaluation.
    runtime = load_frozen_slat30k(device)
    from trellis.modules import sparse as sp
    parent_meta = torch.load(runtime['binding']['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    train_uids = set(parent_meta['data_identity']['object_uids']); del parent_meta
    atomic_json(root/f'binding_{args.start}_{args.end}.json', {
        **runtime['binding'], 'cohort_slat_training_overlap': len({r['object_uid'] for r in plan['rows']} & train_uids),
        'scope': 'training-object mesh comparison; not held-out generalization'})
    targets = {}
    rows = [r for r in plan['rows'] if args.start <= r['object_index'] < args.end]
    for row in rows:
        i = row['index']; verify(row['sample']); verify(row['official_slat'])
        sample = select_views(dataset[i], torch.arange(8))
        obj = row['object_index']
        if obj not in targets:
            target_path = root / 'targets' / f'{obj:03d}.ply'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                meta = json.loads(target_path.with_suffix('.json').read_text())
                if meta['identity_hash'] != identity:
                    raise ValueError('target cache identity mismatch')
                target = trimesh.load(verify(meta['mesh']), process=False, force='mesh')
            else:
                with np.load(row['official_slat']['path']) as data:
                    xyz = np.asarray(data['coords'], np.int32)
                    coords = np.concatenate((np.zeros((len(xyz),1), np.int32), xyz), 1)
                    z = sp.SparseTensor(feats=torch.from_numpy(np.asarray(data['feats'], np.float32)).to(device),
                                        coords=torch.from_numpy(coords).to(device))
                target = decode_with_flow_offload(runtime, z); del z
                target.export(target_path)
                atomic_json(target_path.with_suffix('.json'), {'identity_hash': identity, 'mesh': binding(target_path)})
            targets[obj] = target
        target = targets[obj]
        for seed in plan['seeds']:
            master = master_noise(seed, obj, device)
            for case in [c for c in row['cases'] if c['seed'] == seed]:
                mode = case['mode']; name = f'{i:03d}_{seed}_{mode}'
                record_path = root / 'records' / (name+'.json')
                if record_path.exists():
                    saved = json.loads(record_path.read_text())
                    if saved['identity_hash'] != identity or not saved['success']:
                        raise ValueError('mesh record mismatch')
                    verify(saved['mesh']); continue
                atomic_json(root/f'progress_{args.start}_{args.end}.json', {'index': i, 'mode': mode, 'seed': seed})
                begin = time.time()
                with np.load(verify(case['coordinates'])) as data:
                    coords = data['coords']
                if not len(coords):
                    raise RuntimeError('empty SS support; cannot drop failed case')
                runtime['model'].to(device)
                latent, flow = sample_mesh_latent(runtime, sample, sparse_noise(coords, master))
                predicted = decode_with_flow_offload(runtime, latent); del latent
                surface = surface_metrics(predicted, target, count=plan['surface_samples'], seed=seed*1009+obj*9173)
                output = root/'meshes'/(name+'.ply'); output.parent.mkdir(parents=True, exist_ok=True)
                predicted.export(output)
                record = {'identity_hash': identity, 'success': True, 'index': i, 'object_index': obj,
                          'uid': row['uid'], 'object_uid': row['object_uid'], 'domain': row['domain'],
                          'seed': seed, 'mode': mode, 'surface': surface, 'structure': structure_metrics(predicted),
                          'mesh': binding(output), 'flow': flow, 'elapsed_seconds': time.time()-begin}
                atomic_json(record_path, record)
                print(json.dumps({k: record[k] for k in ('index','seed','mode','elapsed_seconds')} |
                                 {'chamfer_l1': surface['chamfer_l1']}), flush=True)
                del predicted; gc.collect(); torch.cuda.empty_cache()
            del master
    atomic_json(root/f'complete_{args.start}_{args.end}.json', {'identity_hash': identity, 'samples': len(rows)})


@torch.no_grad()
def ss(args):
    from ...v2.ss import build_ss
    from ...v2.checkpoints import read_checkpoint, load_state
    from ...v2.layer_predictor import GeometryConfig
    from ...native_ss import LayeredSSCFGFlow, build_layered_ss_components
    from ...checkpoints import load_parent_ss30k, load_target_lari_checkpoint, load_layered_ss_checkpoint
    from ...pretrained import load_trellis_model, DEFAULT_PRETRAINED
    from ...evaluate_ss_occupancy import occupancy_metrics
    device = torch.device('cuda:0')
    dataset = SceneDataset(args.manifest, require_labels=False, require_ss_target=True)
    if args.label.startswith('dev') or args.label.startswith('strong'):
        SceneDataset(SOURCE/'train96.json', require_labels=False).assert_disjoint(dataset)
    indices = [i for i in cohort_indices(args.cohort, len(dataset)) if args.start <= i < args.end]
    for mode in args.modes:
        root = OUT / args.label / mode
        path = original_checkpoint(mode)
        inputs = {'manifest': binding(args.manifest), 'checkpoint': binding(path),
                  'mode': mode, 'code': bind_code(), 'seeds': [42,43], 'ss_steps': 25,
                  'noise_key': 'seed * 1000003 + first8hex(object_uid)', 'cohort': args.cohort,
                  'labels_removed': True}
        if mode == 'v1':
            inputs['checkpoint'] = binding(OLD/'overfit32_ss_predicted_v1/checkpoints/step_002000.pt')
            inputs['geometry'] = binding(OLD/'overfit32_lari_v1/checkpoints/step_002000.pt')
        identity = canonical_json_sha256(inputs)
        identity_path = root/f'identity_{args.start}_{args.end}.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['identity_hash'] != identity:
            raise ValueError('SS evaluation identity changed')
        atomic_json(identity_path, {'identity_hash': identity, 'identity': inputs})
        if mode == 'v1':
            sampler, model, _ = build_layered_ss_components(device=device, gradient_checkpointing=False)
            parent = load_parent_ss30k(model)
            lari = load_target_lari_checkpoint(model.target_lari, inputs['geometry']['path'])
            load_layered_ss_checkpoint(model, inputs['checkpoint']['path'], parent_binding=parent,
                                      target_lari_binding=lari, layer_source='predicted')
        else:
            checkpoint = read_checkpoint(path); saved = checkpoint['args']
            config = GeometryConfig(**checkpoint['geometry_config'])
            sampler, model, _binding = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'], False,
                                               absolute_tolerance=saved.get('ss_absolute_tolerance'))
            if mode != 'parent':
                load_state(model, checkpoint, config, stage='ss', bindings=_binding, ema=mode == 'ema')
            del checkpoint
        decoder, _ = load_trellis_model(DEFAULT_PRETRAINED, 'ss_dec_conv3d_16l8_fp16')
        decoder.to(device).eval().requires_grad_(False); model.eval()
        for i in indices:
            original = dataset[i]
            sample = select_views({k: original[k] for k in KEYS if k in original}, torch.arange(len(original['intrinsics'])))
            positive = sample['stock_condition'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                conditions = model.build_conditions(sample, device=device, use_layer_branch=mode != 'parent')
            for seed in (42,43):
                record_path = root/'records'/f'{i:03d}_{seed}.json'
                if record_path.exists():
                    previous = json.loads(record_path.read_text())
                    if previous['identity_hash'] != identity:
                        raise ValueError('SS record identity changed')
                    verify(previous['coordinates']); continue
                begin = time.time()
                generator = torch.Generator(device=device).manual_seed(seed*1000003+int(original['object_uid'][:8],16))
                noise = torch.randn((1,8,16,16,16), device=device, generator=generator)
                flow = LayeredSSCFGFlow(model, positive, sample, use_layer_branch=mode != 'parent')
                flow.cached_conditions = conditions[:2]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = sampler.sample(flow, noise, cond=positive, neg_cond=torch.zeros_like(positive),
                        steps=25, cfg_strength=5., cfg_interval=[.5,1.], rescale_t=3., verbose=False).samples
                    logits = decoder(latent.to(next(decoder.parameters()).dtype))
                if not torch.isfinite(logits).all() or not torch.isfinite(latent).all():
                    raise FloatingPointError('nonfinite SS generation')
                coords = torch.argwhere(logits > 0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
                coord_path = root/'coordinates'/f'{i:03d}_{seed}.npz'; coord_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(coord_path, coords=coords)
                record = {'identity_hash': identity, 'index': i, 'object_uid': original['object_uid'],
                          'uid': original['uid'], 'domain': original['data_domain'], 'mode': mode, 'seed': seed,
                          **occupancy_metrics(coords, original['target_coords'].numpy()),
                          'target_support_mean': float(conditions[2].get('target_support_mean', 0.)),
                          'coordinates': binding(coord_path), 'elapsed_seconds': time.time()-begin}
                atomic_json(record_path, record)
                print(json.dumps({k: record[k] for k in ('index','mode','seed','iou','elapsed_seconds')}), flush=True)
        del model, sampler, decoder; gc.collect(); torch.cuda.empty_cache()
    atomic_json(OUT/args.label/f'complete_{args.start}_{args.end}_{"_".join(args.modes)}.json', {'indices': indices, 'modes': args.modes})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare_mesh','mesh','ss'))
    p.add_argument('--gpu', type=int, choices=range(8), default=0)
    p.add_argument('--start', type=int, default=0); p.add_argument('--end', type=int, default=96)
    p.add_argument('--manifest', default=str(OLD/'data_v2/dev64.json'))
    p.add_argument('--label', default='dev64')
    p.add_argument('--modes', nargs='+', default=['parent','v1','raw'],
                   choices=['parent','v1','raw','ema','baseline500',*ARMS])
    p.add_argument('--cohort', choices=('all','paired32'), default='all')
    a = p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2)
    if a.phase == 'prepare_mesh': prepare_mesh()
    elif a.phase == 'mesh': mesh(a)
    else: ss(a)


if __name__ == '__main__':
    main()
