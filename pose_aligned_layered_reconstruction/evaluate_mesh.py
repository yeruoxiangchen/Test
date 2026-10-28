"""Immutable Dev64 SS-to-frozen-SLat30K paired mesh evaluation.

GT is the official SLat decoded by the stock mesh decoder, NOT an original
uncompressed source mesh. Never filter failures or align output meshes to GT.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import trimesh

from .frozen_slat import (DEFAULT_SLAT, SLAT_SHA256, canonical_coords, load_frozen_slat30k,
    master_noise, noise_features, sparse_noise, sample_mesh_latent)
from .mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from .paired_dataset import PairedLayerDataset, sha256_file, canonical_json_sha256
from .prepare_official_experiment import atomic_json, read_json
from .evaluate_ss_occupancy import occupancy_metrics


PACKAGE = Path(__file__).resolve().parent
ROUND1 = PACKAGE / 'outputs/round1_20260909'
EXPERIMENT_ROOT = PACKAGE / 'outputs/mesh_comparison_20260910'
DEFAULT_OUTPUT = EXPERIMENT_ROOT / 'dev64_v2'
DEV_SHA = 'b286abab5ba0908e8ad4edca423bb3309c5694b84cc889e9fb37fb1a358fd3c0'
SEEDS = (42, 43)
MODES = ('parent', 'candidate')


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path))


def verify_file(item):
    if sha256_file(item['path']) != item['sha256']:
        raise RuntimeError('input/output content changed: ' + item['path'])
    return Path(item['path'])


def code_binding():
    # Exclude validation-only legacy oracle and generated outputs.
    own = {str(p.relative_to(PACKAGE)): sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}
    external = PACKAGE.parent/'ReconViaGen/trellis'
    own.update({'third_party/trellis/'+str(p.relative_to(external)): sha256_file(p) for p in sorted(external.rglob('*.py'))})
    return own


def load_plan(output):
    value = read_json(Path(output) / 'plan.json')
    plan = value['plan']
    if canonical_json_sha256(plan) != value['identity_hash']:
        raise RuntimeError('mesh plan identity differs')
    if code_binding() != plan['code_sha256']:
        raise RuntimeError('production source changed after freezing mesh experiment')
    verify_file(plan['manifest'])
    verify_file(plan['slat_checkpoint'])
    return plan, value['identity_hash']


def prepare(output, reference_report):
    output = Path(output).resolve()
    if (output / 'plan.json').exists():
        raise FileExistsError('mesh plan already frozen; reuse without re-preparing')
    reference = read_json(reference_report)
    if not reference.get('passed') or not reference.get('sampling', {}).get('bit_exact'):
        raise RuntimeError('complete 25-step reference equivalence is required')
    if reference['binding']['sha256'] != SLAT_SHA256:
        raise RuntimeError('reference validation used another SLat checkpoint')
    for name, digest in reference['runtime_code_sha256'].items():
        if sha256_file(PACKAGE/name) != digest:
            raise RuntimeError('SLat runtime changed after reference validation')
    manifest = ROUND1 / 'data_v2/dev64.json'
    if sha256_file(manifest) != DEV_SHA:
        raise RuntimeError('not the first-round frozen Dev64')
    dataset = PairedLayerDataset(manifest, require_labels=False, verify_hashes=True)
    if len(dataset) != 64:
        raise RuntimeError('Dev64 object count differs')
    ss_root = ROUND1 / 'formal_dev64_eval'
    identities = [read_json(p) for p in sorted(ss_root.glob('identity_*.json'))]
    if not identities or any(v != identities[0] for v in identities):
        raise RuntimeError('SS evaluation shards have different identities')
    ss_identity = identities[0]
    if (ss_identity['manifest_sha256'] != DEV_SHA or ss_identity['mode'] != 'paired'
        or ss_identity['seeds'] != '42,43' or ss_identity['layer_source'] != 'predicted'
        or not ss_identity['ground_truth_removed_from_predicted_condition']):
        raise RuntimeError('SS evaluation protocol differs')
    for name, expected_sha, expected_step in (
        ('parent', '042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f', 30000),
        ('candidate', '24fb006c1eff78ccf4ceeae67c8fdb417357b4dc8a28e6110baf80b1379902d3', 3000),
        ('lari', 'b7dc68a32631378eeafaf251850cd5ac4a0d50cfad26e108c4ca25d0b052dd40', 4000)):
        value = ss_identity[name]
        verify_file(value)
        if value['sha256'] != expected_sha or value['step'] != expected_step:
            raise RuntimeError('unexpected first-round checkpoint: ' + name)
        if name != 'parent' and set(dataset.identity()['object_uids']) & set(value['data_identity']['object_uids']):
            raise RuntimeError('LaRI/SS adapter training/evaluation overlap')
    ss_hash = canonical_json_sha256(ss_identity)
    rows = []
    for index, row in enumerate(dataset.rows):
        sample_path = dataset._resolve(row['sample'])
        sample = torch.load(sample_path, map_location='cpu', mmap=True, weights_only=True)
        if sample['uid'] != row['uid']:
            raise RuntimeError('sample UID mismatch')
        gt = sample['target_coords'].numpy()
        slat_source = dict(path=row['source']['slat'], sha256=row['source']['slat_sha256'])
        verify_file(slat_source)
        with np.load(slat_source['path']) as payload:
            source_coords = canonical_coords(payload['coords'])
            if not np.array_equal(source_coords, canonical_coords(gt)):
                raise RuntimeError('official SLat support / SS target mismatch')
        cases = []
        for seed in SEEDS:
            for mode in MODES:
                name = f'{index:05d}_{seed}_{mode}'
                record_path = ss_root / 'records' / (name + '.json')
                record = read_json(record_path)
                if record['identity_hash'] != ss_hash or len(record['records']) != 1:
                    raise RuntimeError('SS record identity mismatch')
                old = record['records'][0]
                if (old['index'], old['uid'], old['seed'], old['mode']) != (index, row['uid'], seed, mode):
                    raise RuntimeError('SS record identity/filename mismatch')
                coord_path = ss_root / 'coordinates' / (name + '.npz')
                with np.load(coord_path) as payload:
                    if str(payload['uid']) != row['uid']:
                        raise RuntimeError('SS coordinates UID mismatch')
                    coords = canonical_coords(payload['coords'])
                measured = occupancy_metrics(coords, gt)
                if any(abs(measured[k] - old[k]) > 1.e-12 for k in measured):
                    raise RuntimeError('saved SS coordinates do not reproduce the occupancy record')
                cases.append(dict(seed=seed, mode=mode, name=name, coordinates=binding(coord_path),
                                  ss_record=binding(record_path), ss_metrics=measured))
        rows.append(dict(index=index, uid=row['uid'], sample=dict(path=str(sample_path), sha256=row['sha256']),
                         official_slat=slat_source, expected_decoder=row['decoder_binding'], cases=cases))
        print(json.dumps(dict(prepared_mesh_objects=index+1, total=64)), flush=True)
    plan = dict(format='palr.frozen_slat30k_paired_mesh.v1', manifest=binding(manifest),
        slat_checkpoint=binding(DEFAULT_SLAT), reference_validation=binding(reference_report),
        ss_identity=ss_identity, ss_identity_hash=ss_hash, rows=rows, seeds=list(SEEDS),
        target='official_SLat_raw_features_decoded_with_frozen_stock_mesh_decoder_no_denormalization',
        target_is_original_source_mesh=False, output_frame='native_sparse_grid_identity',
        per_mesh_alignment=False, per_mesh_rescaling=False, mesh_refinement=False,
        common_coordinate_noise='historical_dense64x64x64x8_master_seed*2000003+index*2017+7919',
        sampler=reference['binding']['sampler'], amp='bf16_flow_decoder_native_dtype',
        determinism=reference['binding']['determinism'],
        memory_policy='flow_CPU_offload_during_unchanged_GPU_mesh_decode',
        cuda_allocator='expandable_segments:True',
        surface_samples=20000, surface_seed='seed*1009+index*9173', thresholds=[.01,.02,.05],
        failures='fail_closed_no_case_filtering_no_success_only_aggregate', code_sha256=code_binding())
    if plan['slat_checkpoint']['sha256'] != SLAT_SHA256:
        raise RuntimeError('SLat checkpoint hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'plan.json', dict(identity_hash=canonical_json_sha256(plan), plan=plan))
    print('Frozen mesh plan: 64 objects x 2 seeds x 2 SS models = 256 meshes', flush=True)


def save_mesh_artifacts(mesh, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return binding(path)


def read_completed(path, identity_hash):
    saved = read_json(path)
    if saved.get('identity_hash') != identity_hash or not saved.get('success'):
        raise RuntimeError('existing mesh record is incompatible or failed')
    for key in ('mesh', 'latent', 'target_mesh'):
        verify_file(saved[key])
    return saved


def decode_with_flow_offload(runtime, latent):
    """Change residency only: keep all decoder operators, precision and cubes."""
    runtime['model'].cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)


@torch.no_grad()
def worker(output, indices, worker_name):
    output = Path(output).resolve()
    plan, identity_hash = load_plan(output)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', plan['cuda_allocator'])
    if os.environ['PYTORCH_CUDA_ALLOC_CONF'] != plan['cuda_allocator']:
        raise RuntimeError('CUDA allocator differs from frozen memory policy')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    runtime = load_frozen_slat30k(device, evaluation_uids=[r['uid'] for r in plan['rows']])
    from trellis.modules import sparse as sp
    atomic_json(output / 'workers' / f'{worker_name}_binding.json', runtime['binding'])
    progress = dict(state='running', indices=indices, current_index=None, completed=0,
                    expected=len(indices)*4, identity_hash=identity_hash, pid=os.getpid())
    status_path = output / 'workers' / f'{worker_name}_status.json'
    atomic_json(status_path, progress)
    begin = time.time()
    try:
        for index in indices:
            row = plan['rows'][index]
            progress.update(current_index=index, uid=row['uid'], phase='load_sample')
            atomic_json(status_path, progress)
            verify_file(row['sample'])
            verify_file(row['official_slat'])
            if row['expected_decoder'] != runtime['binding']['decoder']:
                raise RuntimeError('target labelling and mesh evaluation decoders differ')
            sample = torch.load(row['sample']['path'], map_location='cpu', weights_only=True)
            target_path = output / 'targets' / f'{index:05d}.ply'
            target_meta = target_path.with_suffix('.json')
            if target_meta.exists():
                target_info = read_json(target_meta)
                if target_info['identity_hash'] != identity_hash or target_info['uid'] != row['uid']:
                    raise RuntimeError('cached target identity differs')
                target = trimesh.load(verify_file(target_info['mesh']), process=False, force='mesh')
            else:
                progress['phase'] = 'decode_official_target'
                atomic_json(status_path, progress)
                with np.load(row['official_slat']['path']) as payload:
                    coords = np.asarray(payload['coords'])
                    feats = np.asarray(payload['feats'], np.float32)
                    canonical_coords(coords)  # validate support; preserve coord/feature row pairing
                    if coords.shape != (len(feats), 3) or feats.shape != (len(coords), 8) or not np.isfinite(feats).all():
                        raise RuntimeError('invalid official SLat coords/features')
                    coords = coords.astype(np.int32)
                latent = sp.SparseTensor(feats=torch.from_numpy(feats).to(device),
                    coords=torch.from_numpy(np.concatenate((np.zeros((len(coords),1), np.int32), coords),1)).to(device))
                # Official features are already unnormalized; do not mean/std twice.
                target = decode_with_flow_offload(runtime, latent)
                del latent
                target_info = dict(identity_hash=identity_hash, uid=row['uid'],
                    mesh=save_mesh_artifacts(target, target_path), structure=structure_metrics(target))
                atomic_json(target_meta, target_info)
            for seed in SEEDS:
                cases = {c['mode']: c for c in row['cases'] if c['seed'] == seed}
                supports = {}
                for mode in MODES:
                    with np.load(verify_file(cases[mode]['coordinates'])) as payload:
                        supports[mode] = canonical_coords(payload['coords'])
                master = master_noise(seed, index, device)
                # Independent indexing from both branches must agree on intersection.
                flat = {m: c[:,1].astype(np.int64)*4096+c[:,2]*64+c[:,3] for m,c in supports.items()}
                common, left, right = np.intersect1d(flat['parent'], flat['candidate'], return_indices=True)
                a, b = noise_features(supports['parent'], master), noise_features(supports['candidate'], master)
                if not torch.equal(a[torch.as_tensor(left, device=device)], b[torch.as_tensor(right, device=device)]):
                    raise RuntimeError('shared-coordinate noise is not bit exact')
                noise_audit = dict(common_coordinate_count=len(common), common_coordinate_noise_bit_exact=True)
                del a, b
                # Balance branch order without looking at any evaluation score.
                order = MODES if (index+seed) % 2 == 0 else tuple(reversed(MODES))
                for mode in order:
                    case = cases[mode]
                    path = output / 'records' / (case['name'] + '.json')
                    if path.exists():
                        read_completed(path, identity_hash)
                        progress['completed'] += 1
                        continue
                    progress.update(phase='sample_slat', seed=seed, mode=mode)
                    atomic_json(status_path, progress)
                    start = time.time()
                    coords = supports[mode]
                    if not len(coords):
                        raise RuntimeError('empty SS support: failed mesh, never silently excluded')
                    torch.cuda.reset_peak_memory_stats(device)
                    runtime['model'].to(device)
                    latent, flow_summary = sample_mesh_latent(runtime, sample, sparse_noise(coords, master))
                    latent_path = output / 'latents' / (case['name'] + '.npz')
                    latent_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(latent_path, coords=latent.coords.cpu().numpy(),
                                        feats=latent.feats.cpu().numpy(), uid=row['uid'])
                    progress['phase'] = 'decode_mesh'
                    atomic_json(status_path, progress)
                    torch.cuda.empty_cache()
                    mesh = decode_with_flow_offload(runtime, latent)
                    del latent
                    structure = structure_metrics(mesh)
                    progress['phase'] = 'surface_metrics'
                    atomic_json(status_path, progress)
                    surface = surface_metrics(mesh, target, count=plan['surface_samples'],
                        seed=seed*1009+index*9173, thresholds=plan['thresholds'])
                    record = dict(identity_hash=identity_hash, success=True, index=index, uid=row['uid'],
                        seed=seed, mode=mode, surface=surface, structure=structure, ss_metrics=case['ss_metrics'],
                        mesh=save_mesh_artifacts(mesh, output / 'meshes' / (case['name']+'.ply')),
                        latent=binding(latent_path), target_mesh=target_info['mesh'], noise_audit=noise_audit,
                        flow=flow_summary, elapsed_seconds=time.time()-start,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                    atomic_json(path, record)
                    progress['completed'] += 1
                    atomic_json(status_path, progress)
                    print(json.dumps(dict(index=index, seed=seed, mode=mode, completed=progress['completed'],
                        expected=progress['expected'], cd=surface['chamfer_l1'], fscore=surface['fscore_0p02'],
                        seconds=record['elapsed_seconds'])), flush=True)
                    del mesh
                    gc.collect()
                    torch.cuda.empty_cache()
                del master
            del sample, target
        progress.update(state='complete', elapsed_seconds=time.time()-begin)
        atomic_json(status_path, progress)
    except Exception as error:
        progress.update(state='failed', error=repr(error), elapsed_seconds=time.time()-begin)
        atomic_json(status_path, progress)
        raise


def aggregate(output):
    output = Path(output).resolve()
    plan, identity_hash = load_plan(output)
    records = [read_completed(p, identity_hash) for p in sorted((output/'records').glob('*.json'))]
    report = aggregate_mesh_records(records, [r['uid'] for r in plan['rows']], SEEDS)
    report.update(identity_hash=identity_hash, target=plan['target'], target_is_original_source_mesh=False,
                  record_files=[str(p) for p in sorted((output/'records').glob('*.json'))])
    atomic_json(output / 'aggregate.json', report)
    print(json.dumps(report['summary'], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'aggregate'))
    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--reference_report', default=str(EXPERIMENT_ROOT/'smoke_reference_v4/report.json'))
    parser.add_argument('--indices', default='0')
    parser.add_argument('--worker_name', default='smoke')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output_dir, args.reference_report)
    elif args.action == 'aggregate':
        aggregate(args.output_dir)
    else:
        indices = [int(v) for v in args.indices.split(',')]
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= 64:
            raise ValueError('invalid mesh shard indices')
        if not args.worker_name.replace('_', '').isalnum():
            raise ValueError('invalid worker name')
        worker(args.output_dir, indices, args.worker_name)


if __name__ == '__main__':
    main()
