"""Frozen Dev64 / strong16 mesh evaluation; resumable and object-paired.

Only this new output tree is written. The 128 historical clean parent meshes and
64 official-SLat proxy targets are reused after content/protocol checks. New
meshes are exported then reloaded before scoring, without pose/scale fitting.
"""
from __future__ import annotations

import os
for _key, _value in {
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'SPCONV_ALGO': 'native',
    'CUDA_HOME': '/usr/local/cuda-12.6', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ.setdefault(_key, _value)

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ...paired_dataset import sha256_file, canonical_json_sha256
from ...prepare_official_experiment import atomic_json
from ...v2.dataset import SceneDataset, select_views
from ...frozen_slat import DEFAULT_SLAT, SAMPLER_PARAMS, canonical_coords

PACKAGE = Path(__file__).resolve().parents[2]
DIAGNOSTICS = PACKAGE/'outputs/round2_v2_diagnostics_20261018'
SOURCE = PACKAGE/'outputs/round2_v2_overfit32_20261017'
HISTORICAL = PACKAGE/'outputs/mesh_comparison_20260910/dev64_v2'
DEFAULT_OUTPUT = PACKAGE/'outputs/round2_v2_heldout_mesh_20261018'
MANIFESTS = {
    'clean': PACKAGE/'outputs/round1_20260909/data_v2/dev64.json',
    'strong_occluded': DIAGNOSTICS/'strong_data/manifest.json',
}
SEEDS = (42, 43)
MODES = ('parent', 'raw')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def verify(item):
    if sha256_file(item['path']) != item['sha256']:
        raise ValueError('content changed: ' + item['path'])
    return Path(item['path'])


def code_binding():
    from ...evaluate_mesh import code_binding as runtime_code
    result = runtime_code()
    for path in [Path(__file__), PACKAGE/'v2/dataset.py', PACKAGE/'v2/contracts.py']:
        result[str(path.relative_to(PACKAGE))] = sha256_file(path)
    return result


def load_plan(output, check_code=True):
    payload = read(Path(output)/'plan.json')
    if canonical_json_sha256(payload['plan']) != payload['identity_hash']:
        raise ValueError('plan identity mismatch')
    if check_code and code_binding() != payload['plan']['code']:
        raise ValueError('code changed after freezing this experiment')
    return payload['plan'], payload['identity_hash']


def case_name(row, case):
    return f"{row['index']:03d}_{case['seed']}_{case['mode']}"


def _sample_path(dataset, row):
    return dataset.base._resolve(row['sample']) if dataset.base else dataset.root/row['sample']


def _coords(path):
    with np.load(path, allow_pickle=False) as data:
        return canonical_coords(data['coords'])


def _training_audit(evaluation_uids):
    candidate = SOURCE/'ss/checkpoints/step_002000.pt'
    checkpoint = torch.load(candidate, map_location='cpu', mmap=True, weights_only=False)
    parent = Path(checkpoint['bindings']['parent']['path'])
    del checkpoint
    paths = {'parent_ss30k': parent, 'frozen_slat30k': DEFAULT_SLAT,
             'v2_ss_step2000': candidate, 'v2_lari_step2000': SOURCE/'lari/checkpoints/step_002000.pt'}
    result = {}
    for name, path in paths.items():
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        training = set(saved['data_identity']['object_uids'])
        overlap = sorted(training & set(evaluation_uids))
        if overlap:
            raise ValueError(f'{name} train/evaluation overlap: {overlap}')
        result[name] = {'checkpoint': binding(path), 'training_object_count': len(training),
                        'training_uid_hash': canonical_json_sha256(sorted(training)),
                        'evaluation_object_count': len(evaluation_uids), 'overlap': overlap,
                        'step': saved['step'], 'disjoint': True}
        del saved
    return result


def _materialize_reuse(output, plan, identity):
    count = 0
    for row in plan['rows']:
        if row['domain'] != 'clean':
            continue
        for case in row['cases']:
            if case['source_kind'] != 'reused':
                continue
            source = read(verify(case['source_record']))
            verify(source['mesh']); verify(source['target_mesh'])
            record = {**source, 'identity_hash': identity,
                      **{k: row[k] for k in ('index', 'object_index', 'uid', 'object_uid', 'domain')},
                      'source_kind': 'reused', 'source_record': case['source_record'],
                      'ss_record': case['ss_record'], 'coordinates': case['coordinates'],
                      'source_identity_hash': source['identity_hash'],
                      'target_mesh': row['target_mesh'], 'reused_at_utc': plan['created_at_utc']}
            path = output/'records'/(case_name(row, case)+'.json')
            if path.exists():
                old = read(path)
                if old != record:
                    raise ValueError('reused record differs: ' + str(path))
            else:
                atomic_json(path, record)
            count += 1
    if count != 128:
        raise ValueError('expected exactly 128 clean parent reuse records')
    return count


def prepare(args):
    output = Path(args.output).resolve()
    if (output/'plan.json').exists():
        plan, identity = load_plan(output)
        count = _materialize_reuse(output, plan, identity)
        print(json.dumps({'state': 'prepared', 'resumed': True, 'reused_records': count,
                          'identity_hash': identity, 'output': str(output)}), flush=True)
        return
    from ...evaluate_ss_occupancy import occupancy_metrics
    historical_payload = read(HISTORICAL/'plan.json')
    historical = historical_payload['plan']
    if canonical_json_sha256(historical) != historical_payload['identity_hash']:
        raise ValueError('historical plan identity mismatch')
    if (historical['sampler'] != SAMPLER_PARAMS or historical['surface_samples'] != 20000
        or historical['surface_seed'] != 'seed*1009+index*9173'
        or historical['seeds'] != list(SEEDS) or historical['per_mesh_alignment']
        or historical['per_mesh_rescaling']):
        raise ValueError('historical mesh protocol is incompatible')
    # Compare the exact frozen SLat/decoder dependencies, not unrelated later code.
    runtime_files = ('frozen_slat.py', 'evaluate_mesh.py', 'mesh_metrics.py', 'dino_context.py',
                     'projection.py', 'view_fusion.py', 'pretrained.py', '_bootstrap.py')
    for name in runtime_files:
        if sha256_file(PACKAGE/name) != historical['code_sha256'][name]:
            raise ValueError('historical runtime dependency changed: ' + name)
    for name, digest in historical['code_sha256'].items():
        if name.startswith('third_party/trellis/'):
            path = PACKAGE.parent/'ReconViaGen/trellis'/name.removeprefix('third_party/trellis/')
            if sha256_file(path) != digest:
                raise ValueError('historical decoder dependency changed: ' + name)
    datasets = {d: SceneDataset(p, require_labels=False, require_ss_target=True)
                for d, p in MANIFESTS.items()}
    if [len(datasets[d]) for d in ('clean', 'strong_occluded')] != [64, 16]:
        raise ValueError('cohort size changed')
    evaluation_uids = datasets['clean'].identity()['object_uids']
    training_audit = _training_audit(evaluation_uids)
    identity_cache = {}
    for domain, label in (('clean', 'dev64'), ('strong_occluded', 'strong16')):
        for mode in MODES:
            items = []
            for path in sorted((DIAGNOSTICS/label/mode).glob('identity_*.json')):
                item = read(path)
                if canonical_json_sha256(item['identity']) != item['identity_hash']:
                    raise ValueError('SS identity hash mismatch')
                if item['identity']['manifest'] != binding(MANIFESTS[domain]):
                    raise ValueError('SS evaluation uses another manifest')
                if item['identity']['seeds'] != list(SEEDS) or item['identity']['mode'] != mode:
                    raise ValueError('SS mode/seeds mismatch')
                verify(item['identity']['checkpoint'])
                items.append((item['identity_hash'], binding(path)))
            if not items or len({i[0] for i in items}) != 1:
                raise ValueError('SS shards differ')
            identity_cache[domain, mode] = items
    rows, coordinate_checks, camera_audit = [], [], []
    for domain, dataset in datasets.items():
        label = 'dev64' if domain == 'clean' else 'strong16'
        for local_index in range(len(dataset)):
            sample = dataset[local_index]
            obj = local_index
            if sample['object_uid'] != historical['rows'][obj]['uid']:
                raise ValueError('historical object ordering differs')
            target_record = HISTORICAL/'targets'/f'{obj:05d}.json'
            target_info = read(target_record)
            if target_info['identity_hash'] != historical_payload['identity_hash']:
                raise ValueError('historical target identity mismatch')
            verify(target_info['mesh'])
            sample_binding = binding(_sample_path(dataset, dataset.rows[local_index]))
            official = binding(sample['source_binding']['slat'])
            if official != historical['rows'][obj]['official_slat']:
                raise ValueError('official target latent differs')
            if not np.array_equal(_coords(official['path']), canonical_coords(sample['target_coords'].numpy())):
                raise ValueError('SS target and official SLat support differ')
            if domain == 'clean' and sample_binding != historical['rows'][obj]['sample']:
                raise ValueError('clean sample differs from historical parent inference')
            if domain == 'strong_occluded':
                clean = datasets['clean'][obj]
                for key in ('intrinsics', 'extrinsics', 'target_coords'):
                    if not torch.equal(sample[key], clean[key]):
                        raise ValueError('strong/clean geometry differs: ' + key)
                for key in ('grid_transform', 'extrinsics_type', 'camera_forward_sign', 'layer_camera_y_sign', 'image_size'):
                    if sample[key] != clean[key]:
                        raise ValueError('strong/clean camera convention differs: ' + key)
            if sample['grid_transform'] != 'identity' or len(sample['intrinsics']) != 8:
                raise ValueError('unexpected frame/view count')
            rotation = sample['extrinsics'].numpy()[:, :3, :3].astype(np.float64)
            camera_audit.append({'index': obj if domain == 'clean' else 64+obj,
                'object_uid': sample['object_uid'], 'domain': domain, 'grid_transform': sample['grid_transform'],
                'extrinsics_type': sample['extrinsics_type'], 'camera_forward_sign': sample['camera_forward_sign'],
                'layer_camera_y_sign': sample['layer_camera_y_sign'],
                'rotation_max_orthogonality_error': float(np.max(np.abs(rotation@rotation.transpose(0,2,1)-np.eye(3)))),
                'rotation_determinants': np.linalg.det(rotation).tolist(),
                'strong_clean_geometry_bit_exact': True if domain == 'strong_occluded' else None})
            cases = []
            for seed in SEEDS:
                for mode in MODES:
                    ss_path = DIAGNOSTICS/label/mode/'records'/f'{obj:03d}_{seed}.json'
                    ss = read(ss_path)
                    if (ss['identity_hash'] != identity_cache[domain,mode][0][0] or
                        (ss['index'], ss['object_uid'], ss['uid'], ss['seed'], ss['mode']) !=
                        (obj, sample['object_uid'], sample['uid'], seed, mode)):
                        raise ValueError('SS record identity differs')
                    coords = _coords(verify(ss['coordinates']))
                    measured = occupancy_metrics(coords, sample['target_coords'].numpy())
                    if any(abs(measured[k]-ss[k]) > 1.e-12 for k in measured):
                        raise ValueError('SS coordinates do not reproduce stored occupancy')
                    reuse = domain == 'clean' and mode == 'parent'
                    case = {'seed': seed, 'mode': mode, 'coordinates': ss['coordinates'],
                            'ss_record': binding(ss_path), 'source_kind': 'reused' if reuse else 'generated',
                            'ss_metrics': measured}
                    if reuse:
                        old_case = next(c for c in historical['rows'][obj]['cases'] if c['mode'] == mode and c['seed'] == seed)
                        if not np.array_equal(coords, _coords(verify(old_case['coordinates']))):
                            raise ValueError('historical/current parent SS coordinates differ')
                        source_path = HISTORICAL/'records'/f'{obj:05d}_{seed}_parent.json'
                        source = read(source_path)
                        if (not source['success'] or source['identity_hash'] != historical_payload['identity_hash']
                            or source['target_mesh'] != target_info['mesh'] or source['flow'] != {'positive_calls':25,'negative_calls':19}):
                            raise ValueError('historical parent mesh mismatch')
                        verify(source['mesh']); verify(source['latent'])
                        case['source_record'] = binding(source_path)
                        coordinate_checks.append({'object_index': obj, 'seed': seed, 'bit_exact': True})
                    cases.append(case)
            rows.append({'index': obj if domain == 'clean' else 64+obj, 'object_index': obj,
                         'manifest_index': local_index, 'uid': sample['uid'], 'object_uid': sample['object_uid'],
                         'domain': domain, 'sample': sample_binding, 'official_slat': official,
                         'target_mesh': target_info['mesh'], 'target_source_record': binding(target_record),
                         'expected_decoder': historical['rows'][obj]['expected_decoder'], 'cases': cases})
            print(json.dumps({'prepared_inputs': len(rows), 'total': 80}), flush=True)
            del sample
    audit = {'created_at_utc': utc(), 'training_disjointness': training_audit,
             'evaluation_object_uids': evaluation_uids, 'parent_coordinate_equivalence': coordinate_checks,
             'cameras': camera_audit, 'historical_runtime_dependencies_unchanged': True,
             'source_hash_verification': 'all samples, official latent files, SS records/coords, 128 parent meshes/latents and 64 target meshes verified'}
    atomic_json(output/'input_audit.json', audit)
    plan = {'format': 'palr.v2.heldout_mesh.v1', 'created_at_utc': utc(), 'rows': rows,
            'manifests': {d: binding(p) for d,p in MANIFESTS.items()},
            'historical_plan': binding(HISTORICAL/'plan.json'), 'input_audit': binding(output/'input_audit.json'),
            'training_audit': training_audit, 'evaluation_object_uids': evaluation_uids,
            'ss_identities': {d: {m: [x[1] for x in identity_cache[d,m]] for m in MODES} for d in MANIFESTS},
            'modes': list(MODES), 'seeds': list(SEEDS), 'expected_records': 320,
            'expected_new_records': 192, 'expected_reused_records': 128,
            'cohorts': {'clean': {'objects':64,'records':256}, 'strong_occluded': {'objects':16,'records':64}},
            'cohort_policy': 'pre-existing Dev64 and fixed first16 strong set; report separately; Dev64 is development not final untouched test',
            'sampler': SAMPLER_PARAMS, 'surface_samples': 20000, 'surface_seed': 'seed*1009+object_index*9173',
            'thresholds': [.01,.02,.05], 'common_coordinate_noise': 'seed*2000003+object_index*2017+7919 dense64cube',
            'gt': 'official SLat decoded VAE proxy; not original source mesh',
            'native_frame': True, 'per_mesh_alignment': False, 'per_mesh_rescaling': False,
            'surface_scoring': 'new predictions reload exported PLY; reused historical scores independently rechecked by rigid audit',
            'bootstrap': {'unit':'object, average two inference seeds first', 'replicates':5000,'seed':20261018,'uid_order':'sorted'},
            'failures': 'persist failure record; retry allowed; incomplete aggregate forbidden', 'code': code_binding()}
    identity = canonical_json_sha256(plan)
    atomic_json(output/'plan.json', {'plan':plan, 'identity_hash':identity})
    reused = _materialize_reuse(output, plan, identity)
    print(json.dumps({'state':'prepared','output':str(output),'identity_hash':identity,
                      'reused_records':reused,'new_records_required':192}), flush=True)


@torch.no_grad()
def worker(args):
    from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent, noise_features
    from ...evaluate_mesh import decode_with_flow_offload
    from ...mesh_metrics import surface_metrics, structure_metrics
    import trimesh
    output = Path(args.output).resolve()
    plan, identity = load_plan(output)
    verify(plan['input_audit'])
    for item in plan['training_audit'].values():
        verify(item['checkpoint'])
        if item['overlap'] or not item['disjoint']:
            raise ValueError('training overlap in frozen plan')
    rows = [r for r in plan['rows'] if args.start <= r['object_index'] < args.end]
    if args.domain != 'all':
        rows = [r for r in rows if r['domain'] == args.domain]
    name = args.name or f'objects_{args.start}_{args.end}_{args.domain}'
    status_path = output/'workers'/(name+'_status.json')
    status = {'state':'starting','identity_hash':identity,'started_at_utc':utc(),'pid':os.getpid(),
              'object_range':[args.start,args.end], 'domain':args.domain,'generated_this_run':0,
              'completed_or_reused':0,'expected_records_in_range':len(rows)*4}
    atomic_json(status_path,status)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    runtime = load_frozen_slat30k(device, evaluation_uids=plan['evaluation_object_uids'])
    if runtime['binding']['sampler'] != plan['sampler']:
        raise ValueError('runtime sampler mismatch')
    atomic_json(output/'workers'/(name+'_binding.json'), {'identity_hash':identity,**runtime['binding'],
                 'cohort_slat_training_overlap':0,'evaluation_uid_hash':canonical_json_sha256(plan['evaluation_object_uids'])})
    datasets = {d:SceneDataset(verify(item), require_labels=False) for d,item in plan['manifests'].items()}
    try:
        for row in rows:
            verify(row['sample']); verify(row['official_slat'])
            if row['expected_decoder'] != runtime['binding']['decoder']:
                raise ValueError('target/generation decoder mismatch')
            sample = select_views(datasets[row['domain']][row['manifest_index']],torch.arange(8))
            target = trimesh.load(verify(row['target_mesh']),process=False,force='mesh')
            for seed in SEEDS:
                cases = {c['mode']:c for c in row['cases'] if c['seed']==seed}
                supports = {mode:_coords(verify(cases[mode]['coordinates'])) for mode in MODES}
                master = master_noise(seed,row['object_index'],device)
                flat = {m:c[:,1].astype(np.int64)*4096+c[:,2]*64+c[:,3] for m,c in supports.items()}
                common,left,right = np.intersect1d(flat['parent'],flat['raw'],return_indices=True)
                a,b = noise_features(supports['parent'],master),noise_features(supports['raw'],master)
                if not torch.equal(a[torch.as_tensor(left,device=device)],b[torch.as_tensor(right,device=device)]):
                    raise ValueError('shared-coordinate noise differs')
                del a,b
                order = MODES if (row['object_index']+seed)%2==0 else tuple(reversed(MODES))
                for mode in order:
                    case = cases[mode]; item_name = case_name(row,case)
                    record_path = output/'records'/(item_name+'.json')
                    if record_path.exists():
                        saved = read(record_path)
                        if saved['identity_hash'] != identity or not saved['success']:
                            raise ValueError('existing output record mismatch')
                        verify(saved['mesh']); verify(saved['target_mesh'])
                        status['completed_or_reused'] += 1
                        continue
                    if case['source_kind']=='reused':
                        raise ValueError('missing reused record; run prepare recovery')
                    if args.max_new and status['generated_this_run'] >= args.max_new:
                        status.update(state='bounded_complete',updated_at_utc=utc())
                        atomic_json(status_path,status)
                        return
                    status.update(index=row['index'],object_index=row['object_index'],seed=seed,mode=mode,phase='sample_slat',state='running',updated_at_utc=utc())
                    atomic_json(status_path,status)
                    start = time.time()
                    try:
                        coords = supports[mode]
                        if not len(coords):
                            raise ValueError('empty SS support; case cannot be dropped')
                        torch.cuda.reset_peak_memory_stats(device)
                        runtime['model'].to(device)
                        latent,flow = sample_mesh_latent(runtime,sample,sparse_noise(coords,master))
                        predicted = decode_with_flow_offload(runtime,latent); del latent
                        mesh_path = output/'meshes'/(item_name+'.ply')
                        mesh_path.parent.mkdir(parents=True,exist_ok=True)
                        temporary = mesh_path.with_suffix('.partial.ply')
                        predicted.export(temporary); temporary.replace(mesh_path)
                        del predicted
                        predicted = trimesh.load(mesh_path,process=False,force='mesh')
                        surface = surface_metrics(predicted,target,count=plan['surface_samples'],
                            seed=seed*1009+row['object_index']*9173,thresholds=plan['thresholds'])
                        record = {'identity_hash':identity,'success':True,
                            **{k:row[k] for k in ('index','object_index','uid','object_uid','domain')},
                            'seed':seed,'mode':mode,'surface':surface,'structure':structure_metrics(predicted),
                            'mesh':binding(mesh_path),'target_mesh':row['target_mesh'],'flow':flow,
                            'source_kind':'generated','ss_record':case['ss_record'],'coordinates':case['coordinates'],
                            'noise_audit':{'common_coordinate_count':len(common),'common_coordinate_noise_bit_exact':True},
                            'elapsed_seconds':time.time()-start,'completed_at_utc':utc(),
                            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}
                        if flow != {'positive_calls':25,'negative_calls':19}:
                            raise ValueError('SLat CFG call counts differ')
                        atomic_json(record_path,record)
                        status['generated_this_run'] += 1; status['completed_or_reused'] += 1
                        atomic_json(status_path,status)
                        print(json.dumps({k:record[k] for k in ('index','object_index','seed','mode','elapsed_seconds')} |
                                         {'chamfer_l1':surface['chamfer_l1'],'at_utc':utc()}),flush=True)
                        del predicted; gc.collect(); torch.cuda.empty_cache()
                    except Exception:
                        failure = {'identity_hash':identity,'success':False,'case':item_name,
                                   'at_utc':utc(),'traceback':traceback.format_exc()}
                        atomic_json(output/'failures'/f'{item_name}_{time.time_ns()}.json',failure)
                        raise
                del master
            del sample,target
        status.update(state='complete',finished_at_utc=utc())
        atomic_json(status_path,status)
    except Exception:
        status.update(state='failed',updated_at_utc=utc(),traceback=traceback.format_exc())
        atomic_json(status_path,status)
        raise


def paired_summary(records):
    uids = sorted({r['object_uid'] for r in records})
    expected = {(u,s,m) for u in uids for s in SEEDS for m in MODES}
    by_key = {(r['object_uid'],r['seed'],r['mode']):r for r in records}
    if set(by_key)!=expected or len(by_key)!=len(records) or not all(r['success'] for r in records):
        raise ValueError('incomplete/duplicate/failed paired cohort')
    keys = sorted(records[0]['surface'])
    if any(sorted(r['surface'])!=keys or not all(np.isfinite(list(r['surface'].values()))) for r in records):
        raise ValueError('invalid metric schema')
    summary = {m:{k:float(np.mean([r['surface'][k] for r in records if r['mode']==m])) for k in keys} for m in MODES}
    paired = {}; per_object = {u:{} for u in uids}
    for key in keys:
        sign = -1 if key in ('chamfer_l1','chamfer_l2','pred_to_gt_mean','gt_to_pred_mean') else 1
        differences = np.array([[sign*(by_key[u,s,'raw']['surface'][key]-by_key[u,s,'parent']['surface'][key]) for s in SEEDS] for u in uids])
        gains = differences.mean(1)
        paired[key] = effect(gains) | {'by_seed':{str(s):float(differences[:,j].mean()) for j,s in enumerate(SEEDS)}}
        for u,gain in zip(uids,gains):
            per_object[u][key] = float(gain)
    return {'object_count':len(uids),'record_count':len(records),'summary':summary,'paired':paired,
            'per_object_improvements':per_object,'all_meshes_successful':True}


def effect(gains):
    gains = np.asarray(gains,np.float64)
    rng = np.random.default_rng(20261018)
    boot = gains[rng.integers(len(gains),size=(5000,len(gains)))].mean(1)
    return {'mean_improvement':float(gains.mean()),'positive_means_better':True,
            'median_improvement':float(np.median(gains)), 'object_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'win_objects':int((gains>1.e-12).sum()),'loss_objects':int((gains< -1.e-12).sum()),
            'tie_objects':int((np.abs(gains)<=1.e-12).sum())}


def aggregate(args):
    output = Path(args.output).resolve(); plan,identity = load_plan(output)
    expected = {case_name(r,c):(r,c) for r in plan['rows'] for c in r['cases']}
    paths = list((output/'records').glob('*.json'))
    if {p.stem for p in paths} != set(expected):
        raise ValueError(f'incomplete/extra records: found {len(paths)}, expected {len(expected)}')
    records = []
    for path in sorted(paths):
        record = read(path); row,case = expected[path.stem]
        if record['identity_hash']!=identity or not record['success']:
            raise ValueError('mixed/failed records')
        for key in ('index','object_index','uid','object_uid','domain'):
            if record[key]!=row[key]:
                raise ValueError('record/plan mismatch: '+key)
        if (record['seed'],record['mode'])!=(case['seed'],case['mode']):
            raise ValueError('case mismatch')
        verify(record['mesh']); verify(record['target_mesh']); verify(record['ss_record']); verify(record['coordinates'])
        if record['target_mesh']!=row['target_mesh']:
            raise ValueError('wrong target mesh')
        records.append(record)
    cohorts = {d:paired_summary([r for r in records if r['domain']==d]) for d in MANIFESTS}
    strong_uids = sorted(cohorts['strong_occluded']['per_object_improvements'])
    same16 = paired_summary([r for r in records if r['domain']=='clean' and r['object_uid'] in strong_uids])
    interaction = {k:effect([cohorts['strong_occluded']['per_object_improvements'][u][k]-
        same16['per_object_improvements'][u][k] for u in strong_uids]) for k in same16['paired']}
    report = {'identity_hash':identity,'completed_at_utc':utc(),'record_count':len(records),
              'source_counts':{kind:sum(r['source_kind']==kind for r in records) for kind in ('reused','generated')},
              'cohorts':cohorts,'same16_clean':same16,
              'strong_minus_clean_improvement_same16':interaction,
              'interaction_interpretation':'positive means V2-parent improvement is larger under strong occlusion on the same16 objects',
              'bootstrap':plan['bootstrap'],'scope':plan['cohort_policy'],
              'failed_attempt_count':len(list((output/'failures').glob('*.json'))),
              'records_sha256':{p.name:sha256_file(p) for p in sorted(paths)}}
    atomic_json(output/'aggregate.json',report)
    print(json.dumps({'complete':True,'records':len(records),'cohorts':{d:{'means':v['summary'],'cd_effect':v['paired']['chamfer_l1']} for d,v in cohorts.items()}}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','aggregate'))
    parser.add_argument('--output',default=str(DEFAULT_OUTPUT))
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=64)
    parser.add_argument('--name')
    parser.add_argument('--domain',choices=('all','clean','strong_occluded'),default='all')
    parser.add_argument('--max-new',type=int,default=0,help='bounded benchmark; zero means all remaining cases')
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 64 or args.max_new<0:
        parser.error('invalid range/max-new')
    {'prepare':prepare,'worker':worker,'aggregate':aggregate}[args.action](args)


if __name__=='__main__':
    main()
