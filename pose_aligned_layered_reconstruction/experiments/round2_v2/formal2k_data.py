"""Freeze existing clean 2K inputs and optionally resume equal-weight occlusion.

The original caches are read-only. A final training manifest is published only
after every planned sample has passed its content and camera checks; training
must never observe a growing manifest. No GPU is used by ``prepare``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import torch

from ...contracts import LAYERED_SAMPLE_FORMAT
from ...paired_dataset import canonical_json_sha256, sha256_file
from ...v2.contracts import adapt_v1, validate_sample
from ...v2.dataset import SceneDataset, write_manifest

PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / 'outputs/round1_20260909/data_v2'
OLD_OCCLUSION = PACKAGE / 'outputs/round2_v2_overfit32_20261017/occlusion/manifest.json'
DEFAULT_OUTPUT = PACKAGE / 'outputs/round2_v2_formal2k_20261018/data'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def progress(output, payload):
    atomic_json(output / 'progress.json', payload)
    atomic_json(output / 'status.json', payload)


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def verify(item):
    if sha256_file(item['path']) != item['sha256']:
        raise ValueError('changed source: ' + item['path'])
    return Path(item['path'])


def code_binding():
    names = ['experiments/round2_v2/formal2k_data.py',
             'experiments/round2_v2/formal2k_render_audit.py', 'data_v2/build_legacy_occlusion.py',
             'data_v2/build_feature_cache.py', 'data_v2/render_scenes.py',
             'v2/dataset.py', 'v2/contracts.py', 'paired_encoder.py', 'frozen_slat.py',
             'pretrained.py', 'paired_preprocessing.py', 'prepare_official_experiment.py']
    return {name: sha256_file(PACKAGE / name) for name in names}


def row_path(payload, row):
    return (Path(payload['root']) / row['sample']).resolve(strict=True)


def audit_clean(task):
    payload, row = task
    path = row_path(payload, row)
    if sha256_file(path) != row['sha256']:
        raise ValueError('clean cache content differs: ' + str(path))
    sample = torch.load(path, map_location='cpu', weights_only=True)
    if sample['uid'] != row['uid'] or sample['object_uid'] != row['object_uid']:
        raise ValueError('clean UID mismatch')
    if sample['format'] != LAYERED_SAMPLE_FORMAT:
        raise ValueError('unexpected clean cache format')
    validate_sample(adapt_v1(sample), require_labels=True, require_ss_target=True)
    if (tuple(sample['raw_visual_patch_features'].shape) != (8, 1369, 1024)
            or tuple(sample['masked_visual_patch_features'].shape) != (8, 1369, 1024)
            or tuple(sample['layer_depths_gt'].shape) != (8, 5, 37, 37)):
        raise ValueError('formal 2K requires 8 paired views and 5 layers')
    if sample['source_binding'] != row['source']:
        raise ValueError('manifest/cache source binding mismatch')
    source_bindings = {}
    for key in ('cache', 'ss', 'slat'):
        source = sample['source_binding']
        observed = binding(source[key])
        if observed['sha256'] != source[key + '_sha256']:
            raise ValueError('original source content differs: ' + source[key])
        source_bindings[key] = observed
    if not Path(sample['source_binding']['render']).is_file():
        raise FileNotFoundError(sample['source_binding']['render'])
    rotation = sample['extrinsics'][:, :3, :3].float()
    orthogonal_error = float((rotation @ rotation.transpose(-1, -2) - torch.eye(3)).abs().max())
    determinant = torch.linalg.det(rotation)
    if orthogonal_error > 1.e-4 or float((determinant - 1).abs().max()) > 1.e-4:
        raise ValueError('camera extrinsics rotation is not proper')
    return {'uid': row['uid'], 'object_uid': row['object_uid'], 'sample': str(path),
            'sha256': row['sha256'], 'bytes': path.stat().st_size,
            'source_bindings': source_bindings, 'decoder_binding': row['decoder_binding'],
            'camera_max_orthogonal_error': orthogonal_error,
            'camera_min_determinant': float(determinant.min()),
            'camera_max_determinant': float(determinant.max())}


def load_plan(output):
    payload = read(output / 'plan.json')
    if canonical_json_sha256(payload['plan']) != payload['identity_hash']:
        raise ValueError('plan identity mismatch')
    plan = payload['plan']
    if plan['code'] != code_binding():
        raise ValueError('data preparation code changed after freeze')
    verify(plan['clean_source']); verify(plan['dev_source'])
    load_render_audit(plan)
    return plan, payload['identity_hash']


def preserve_incomplete(path):
    """Keep interrupted outputs as evidence; never overwrite their bytes."""
    path = Path(path)
    preserved = path.with_name(path.name + f'.incomplete.{time.time_ns()}')
    path.rename(preserved)
    return preserved


def read_uncommitted_json(path):
    """Only syntax truncation is recoverable before an outer receipt commits it."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = read(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        preserve_incomplete(path)
        return None
    if not isinstance(value, dict):
        raise ValueError('JSON artifact must be an object: ' + str(path))
    return value


def make_manifest(path, rows, config, root):
    path = Path(path)
    actual = read_uncommitted_json(path)
    if actual is not None:
        if (actual['samples'] != rows or actual['config'] != config
                or actual['root'] != str(Path(root).resolve())):
            raise ValueError('existing immutable manifest differs: ' + str(path))
        # Check complete format/count/hash invariants, not just the rows.
        SceneDataset(path, verify_hashes=False)
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    write_manifest(temporary, rows, config, root)
    os.replace(temporary, path)


def publish_prepared_manifests(output, plan, identity):
    """Idempotently complete publication after the plan was atomically frozen."""
    make_manifest(output / 'train_clean2000.json', plan['clean_rows'],
        {'kind': 'formal2k_clean', 'data_plan_identity': identity, 'source': plan['clean_source']}, output)
    make_manifest(output / 'dev64.json', plan['dev_rows'],
        {'kind': 'formal2k_development', 'data_plan_identity': identity, 'source': plan['dev_source']}, output)


def load_render_audit(plan):
    audit_path = verify(plan['selected_render_audit'])
    audit = read(audit_path)
    identity = canonical_json_sha256({key: value for key, value in audit.items() if key != 'identity_hash'})
    if audit['identity_hash'] != identity or audit['passed'] is not True:
        raise ValueError('selected render audit identity or completion differs')
    if audit['binding']['manifest'] != plan['clean_source']:
        raise ValueError('selected render audit binds another clean manifest')
    for path, expected_hash in audit['binding']['code'].items():
        verify({'path': path, 'sha256': expected_hash})
    rows = audit['rows']
    result = {row['uid']: row for row in rows}
    clean_rows = {row['uid']: row for row in plan['clean_rows']}
    if len(result) != len(rows) or len(rows) != audit['count'] or set(result) != set(clean_rows):
        raise ValueError('selected render audit must cover every clean object exactly once')
    for uid, row in result.items():
        reference = clean_rows[uid]
        if (row['object_uid'] != reference['object_uid']
                or row['source_cache_sha256'] != reference['source']['cache_sha256']):
            raise ValueError('selected render audit source identity differs')
        views = row['view_ids']
        hashes = row['selected_render_member_sha256']
        expected_members = {f"{reference['source']['uid']}/{view:03d}{suffix}"
                            for view in views for suffix in ('.rgba.webp', '.json')}
        if (len(views) != 8 or len(set(views)) != 8 or set(hashes) != expected_members
                or any(len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value)
                       for value in hashes.values())):
            raise ValueError('selected render audit has invalid view/member hashes')
    return result


def validate_scene(scene, item, source, plan, render_row):
    recipe = {'seed': item['seed'], 'box_count': plan['box_count'],
              'layers': plan['layers'], 'source_uid': item['source_uid']}
    if (scene['recipe'] != recipe or scene['legacy_source_binding'] != source
            or scene['uid'] != item['uid'] or scene['object_uid'] != item['object_uid']):
        raise ValueError('occlusion scene recipe/source/identity differs')
    if scene['pixel_audit']['selected_render_member_sha256'] != render_row['selected_render_member_sha256']:
        raise ValueError('occlusion source pixels differ from the frozen selected render audit')


def validate_cached_sample(row, item, reference):
    if row['uid'] != item['uid'] or row['object_uid'] != item['object_uid']:
        raise ValueError('occlusion cache row identity differs')
    verify({'path': row['sample'], 'sha256': row['sha256']})
    cached = torch.load(row['sample'], map_location='cpu', weights_only=True)
    validate_sample(cached, require_labels=True, require_ss_target=True)
    if cached['uid'] != item['uid'] or cached['object_uid'] != item['object_uid']:
        raise ValueError('occlusion cache identity differs')
    for key in ('intrinsics', 'extrinsics', 'target', 'target_coords'):
        if not torch.equal(cached[key], reference[key]):
            raise ValueError('occlusion camera/target differs: ' + key)
    for key in ('extrinsics_type', 'camera_forward_sign', 'layer_camera_y_sign', 'grid_transform'):
        if cached[key] != reference[key]:
            raise ValueError('occlusion coordinate convention differs: ' + key)
    if tuple(cached['image_size']) != tuple(reference['image_size']):
        raise ValueError('occlusion image size differs')
    for source in cached['source_files']:
        verify(source)
    scene_path = Path(row['sample']).parent / 'scene.json'
    if binding(scene_path) not in cached['source_files']:
        raise ValueError('occlusion cache does not bind its scene metadata')
    return cached


def verify_completed_receipt(saved, item, identity, folder):
    # A syntactically complete receipt is committed evidence. Identity, content,
    # and missing-field failures stop the run instead of silently rebuilding it.
    if saved['identity_hash'] != identity or saved['item'] != item:
        raise ValueError('completed sample has another preparation identity')
    row = saved['row']
    if (row['uid'] != item['uid'] or row['object_uid'] != item['object_uid']
            or Path(row['sample']).resolve() != (folder / 'sample.pt').resolve()):
        raise ValueError('completed receipt sample identity/path differs')
    verify({'path': row['sample'], 'sha256': row['sha256']})
    verify(saved['scene'])
    if Path(saved['scene']['path']).resolve() != (folder / 'scene.json').resolve():
        raise ValueError('completed receipt scene path differs')
    if saved['scene'] not in saved['source_files'] or saved['camera_target_exact'] is not True:
        raise ValueError('completed receipt lacks camera/scene verification')
    for source in saved['source_files']:
        verify(source)
    return row


def prepare(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        plan, identity = load_plan(output)
        if plan['variants'] != args.variants:
            raise ValueError('cannot change variants while resuming frozen preparation')
        publish_prepared_manifests(output, plan, identity)
        if not plan['variants']:
            finalize(output, plan, identity, [])
        print(json.dumps({'state': 'prepared', 'resumed': True, 'identity_hash': identity}), flush=True)
        return
    render_audit_path = output / 'render_source_audit.json'
    if not render_audit_path.exists():
        raise FileNotFoundError('Run formal2k_render_audit first: ' + str(render_audit_path))
    torch.set_num_threads(1)
    clean_path, dev_path = Path(args.manifest).resolve(), Path(args.dev_manifest).resolve()
    clean_ds, dev_ds = SceneDataset(clean_path), SceneDataset(dev_path)
    clean_ds.assert_disjoint(dev_ds)
    if len(clean_ds) != 2000 or len(dev_ds) != 64:
        raise ValueError('formal cohort must contain clean 2000 and development 64')
    clean, dev = read(clean_path), read(dev_path)
    train_uids = sorted({r['object_uid'] for r in clean['samples']})
    dev_uids = sorted({r['object_uid'] for r in dev['samples']})
    if len(train_uids) != 2000 or len(dev_uids) != 64:
        raise ValueError('formal clean cohorts require one sample per physical object')
    from ...checkpoints import DEFAULT_PARENT_SS30K
    from ...frozen_slat import DEFAULT_SLAT
    parent_audit = {}
    for name, path in [('ss30k', DEFAULT_PARENT_SS30K), ('slat30k', DEFAULT_SLAT)]:
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        uids = set(saved['data_identity']['object_uids'])
        overlap = sorted(uids & set(dev_uids))
        if overlap:
            raise ValueError(f'{name} overlaps development objects')
        parent_audit[name] = {'checkpoint': binding(path), 'training_count': len(uids),
            'training_uid_hash': canonical_json_sha256(sorted(uids)),
            'formal_train_overlap_count': len(uids & set(train_uids)), 'dev_overlap': overlap}
        del saved
    audit_rows = []
    started = time.monotonic()
    tasks = [(payload, row) for payload in (clean, dev) for row in payload['samples']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(audit_clean, tasks), 1):
            audit_rows.append(record)
            if i % 32 == 0 or i == len(tasks):
                progress(output, {'state': 'auditing_clean', 'utc': utc(),
                    'completed': i, 'total': len(tasks), 'elapsed_seconds': time.monotonic() - started})
                print(json.dumps({'stage': 'audit_clean', 'completed': i, 'total': len(tasks)}), flush=True)
    clean_rows = [{**row, 'sample': str(row_path(clean, row)),
                   'source_sample_format': LAYERED_SAMPLE_FORMAT} for row in clean['samples']]
    dev_rows = [{**row, 'sample': str(row_path(dev, row)),
                 'source_sample_format': LAYERED_SAMPLE_FORMAT} for row in dev['samples']]
    render_binding = binding(render_audit_path)
    render_rows = load_render_audit({'selected_render_audit': render_binding,
        'clean_source': binding(clean_path), 'clean_rows': clean_rows})
    expected, reused = [], []
    old_payload = read(OLD_OCCLUSION) if args.variants and OLD_OCCLUSION.exists() else None
    old = {r['uid']: r for r in old_payload['samples']} if old_payload else {}
    for index, row in enumerate(clean['samples']):
        for variant in range(args.variants):
            seed = 20261017 + index * 1009 + variant
            uid = row['uid'] + f':occlusion:{seed}'
            item = {'index': index, 'variant': variant, 'seed': seed, 'uid': uid,
                    'object_uid': row['object_uid'], 'source_uid': row['uid']}
            if uid in old:
                cached_row = old[uid]
                path = row_path(old_payload, cached_row)
                if sha256_file(path) != cached_row['sha256']:
                    raise ValueError('historical occlusion cache changed')
                scene = read(path.parent / 'scene.json')
                expected_recipe = {'seed': seed, 'box_count': 2, 'layers': 5, 'source_uid': row['uid']}
                if scene['recipe'] != expected_recipe or scene['legacy_source_binding'] != row['source']:
                    raise ValueError('historical occlusion recipe/source differs')
                validate_scene(scene, item, row['source'], {'box_count': 2, 'layers': 5}, render_rows[row['uid']])
                sample = torch.load(path, map_location='cpu', weights_only=True)
                validate_sample(sample, require_labels=True, require_ss_target=True)
                reference = torch.load(row_path(clean, row), map_location='cpu', weights_only=True)
                for key in ('intrinsics', 'extrinsics', 'target', 'target_coords'):
                    if not torch.equal(sample[key], reference[key]):
                        raise ValueError('historical occlusion camera/target differs: ' + key)
                for source in sample['source_files']:
                    verify(source)
                del sample, reference
                item['reuse'] = {**cached_row, 'sample': str(path)}
                reused.append(uid)
            expected.append(item)
    decoder_bindings = {canonical_json_sha256(r['decoder_binding']): r['decoder_binding'] for r in audit_rows}
    if len(decoder_bindings) != 1:
        raise ValueError('clean input records bind different proxy decoders')
    dino_weights = Path(torch.hub.get_dir()) / 'checkpoints/dinov2_vitl14_reg4_pretrain.pth'
    plan = {'created_at_utc': utc(), 'clean_source': binding(clean_path), 'dev_source': binding(dev_path),
        'variants': args.variants, 'seed': 20261017, 'box_count': 2, 'layers': 5,
        'object_count': 2000, 'clean_count': 2000, 'occluded_count': 2000 * args.variants,
        'final_sample_count': 2000 * (1 + args.variants),
        'scene_sampling': 'one clean and fixed occluded variants per object; uniform rows',
        'train_uids': train_uids, 'dev_uids': dev_uids, 'parent_audit': parent_audit,
        'clean_rows': clean_rows, 'dev_rows': dev_rows, 'occluded_rows': expected,
        'old_occlusion_source': binding(OLD_OCCLUSION) if old_payload else None,
        'proxy_decoder_binding': next(iter(decoder_bindings.values())),
        'dino_weights': binding(dino_weights), 'selected_render_audit': render_binding,
        'reused_occlusion_count': len(reused), 'code': code_binding(),
        'limitations': ['train2000 overlaps parent training; this is adaptation, not novel training objects',
            'Dev64 was previously inspected and is a development cohort, not an untouched final test',
            'layer labels use frozen official SLat VAE proxy geometry, not original high-resolution GT',
            'public backbone pretraining overlap is not certified',
            'legacy clean cap-at-five rays are conservatively marked censored by adapt_v1']}
    identity = canonical_json_sha256(plan)
    atomic_json(output / 'clean_cache_audit.json', {'created_at_utc': utc(), 'count': len(audit_rows),
        'passed': True, 'elapsed_seconds': time.monotonic() - started, 'rows': audit_rows})
    atomic_json(output / 'plan.json', {'plan': plan, 'identity_hash': identity})
    publish_prepared_manifests(output, plan, identity)
    progress(output, {'state': 'prepared', 'utc': utc(), 'identity_hash': identity,
        'reused_occlusion_count': len(reused), 'remaining_occlusion_count': len(expected) - len(reused)})
    if not args.variants:
        finalize(output, plan, identity, [])
    print(json.dumps({'state': 'prepared', 'identity_hash': identity, 'variants': args.variants,
                      'reused': len(reused), 'new': len(expected) - len(reused)}), flush=True)


def finalize(output, plan, identity, rows):
    if len(rows) != plan['occluded_count']:
        raise ValueError('refusing to publish incomplete training manifest')
    all_rows = plan['clean_rows'] + rows
    if len({r['uid'] for r in all_rows}) != plan['final_sample_count']:
        raise ValueError('missing or duplicate formal samples')
    counts = {}
    for row in all_rows:
        counts[row['object_uid']] = counts.get(row['object_uid'], 0) + 1
    if set(counts) != set(plan['train_uids']) or set(counts.values()) != {1 + plan['variants']}:
        raise ValueError('unequal physical-object sampling weights')
    if set(counts) & set(plan['dev_uids']):
        raise ValueError('training/development overlap')
    if rows:
        import numpy as np
        fractions, scene_means, valid_fractions = [], [], []
        for row in rows:
            scene = read(Path(row['sample']).parent / 'scene.json')
            values = scene['visible_fraction']
            if len(values) != 8 or not all(0 <= value <= 1 for value in values):
                raise ValueError('invalid per-view visibility record')
            fractions.extend(values)
            scene_means.append(float(np.mean(values)))
            valid_fraction = scene['supervision_valid_fraction']
            if not 0 <= valid_fraction <= 1:
                raise ValueError('invalid supervision valid fraction')
            valid_fractions.append(valid_fraction)
        histogram, edges = np.histogram(fractions, bins=[0, .02, .25, .5, .75, 1.000001])
        atomic_json(output / 'visibility_summary.json', {'identity_hash': identity,
            'scene_count': len(rows), 'view_count': len(fractions),
            'selection_policy': 'fixed random 3D boxes; no visibility or reconstruction-based filtering',
            'visible_fraction_mean': float(np.mean(fractions)),
            'visible_fraction_quantiles': np.quantile(fractions, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            'visible_fraction_bin_edges': edges.tolist(), 'visible_fraction_bin_counts': histogram.tolist(),
            'per_scene_mean_quantiles': np.quantile(scene_means, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            'supervision_valid_fraction_mean': float(np.mean(valid_fractions))})
    target = output / f"train{plan['final_sample_count']}.json"
    make_manifest(target, all_rows, {'kind': 'formal2k_fixed_equal_weight_mixture',
        'data_plan_identity': identity, 'clean_source': plan['clean_source'],
        'variants_per_object': plan['variants'], 'samples_per_object': 1 + plan['variants'],
        'target_geometry': 'official_SLat_VAE_proxy', 'development_object_count': len(plan['dev_uids'])}, output)
    progress(output, {'state': 'complete', 'utc': utc(), 'identity_hash': identity,
        'manifest': binding(target), 'sample_count': len(all_rows), 'object_count': len(counts),
        'reused_occlusion_count': plan['reused_occlusion_count'], 'generated_occlusion_count':
            plan['occluded_count'] - plan['reused_occlusion_count']})


def augment_data(args):
    output = Path(args.output_dir).resolve()
    plan, identity = load_plan(output)
    publish_prepared_manifests(output, plan, identity)
    render_rows = load_render_audit(plan)
    if not plan['variants']:
        finalize(output, plan, identity, [])
        return
    from ...data_v2.build_legacy_occlusion import augment
    from ...data_v2.build_feature_cache import cache_scene
    from ...paired_encoder import build_frozen_dino_encoder
    from ...frozen_slat import load_frozen_slat30k
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('occlusion rasterization requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    verify(plan['dino_weights'])
    runtime = load_frozen_slat30k(device)
    if runtime['binding']['decoder'] != plan['proxy_decoder_binding']:
        raise ValueError('GPU decoder differs from the clean-data and frozen recipe decoder')
    # Augmentation decodes an existing official SLat target; the SLat flow is
    # never called. Keep its audited weights on CPU to reserve VRAM for large GT.
    runtime['model'].cpu()
    torch.cuda.empty_cache()
    encoder = build_frozen_dino_encoder(device=device)
    dataset = SceneDataset(plan['clean_source']['path'], require_ss_target=True)
    completed, started, previous_index, sample = [], time.monotonic(), None, None
    atomic_json(output / 'runtime_binding.json', {'created_at_utc': utc(), 'identity_hash': identity,
        'runtime': runtime['binding'], 'device': str(device), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'unused_frozen_slat_flow_offloaded_to_cpu': True,
        'pid': os.getpid(), 'code': code_binding()})
    for item in plan['occluded_rows']:
        if previous_index != item['index']:
            sample = dataset[item['index']]
            previous_index = item['index']
        if sample['uid'] != item['source_uid'] or sample['object_uid'] != item['object_uid']:
            raise ValueError('occlusion plan source index differs')
        render_row = render_rows[item['source_uid']]
        if 'reuse' in item:
            row = item['reuse']
            cached = validate_cached_sample(row, item, sample)
            scene_path = Path(row['sample']).parent / 'scene.json'
            validate_scene(read(scene_path), item, sample['source_binding'], plan, render_row)
            del cached
        else:
            folder = output / 'occlusion' / f"object_{item['index']:05d}_variant_{item['variant']:02d}"
            scene_path = folder / 'scene.json'
            receipt = output / 'receipts' / f"{item['index']:05d}_{item['variant']:02d}.json"
            saved = read_uncommitted_json(receipt)
            if saved is not None:
                row = verify_completed_receipt(saved, item, identity, folder)
                validate_scene(read(scene_path), item, sample['source_binding'], plan, render_row)
            else:
                # A scene without a completed outer receipt can be interrupted.
                # Preserve malformed/incomplete artifacts before deterministic rebuild.
                scene = read_uncommitted_json(scene_path)
                if folder.exists() and scene is None:
                    preserve_incomplete(folder)
                if scene is None:
                    scene_path = augment(sample, folder, runtime, item['seed'], plan['box_count'], plan['layers'])
                    scene = read(scene_path)
                validate_scene(scene, item, sample['source_binding'], plan, render_row)
                cache_path = folder / 'sample.pt'
                cache_receipt = Path(str(cache_path) + '.receipt.json')
                cache_saved = read_uncommitted_json(cache_receipt)
                if cache_path.exists() and cache_saved is None:
                    preserve_incomplete(cache_path)
                elif not cache_path.exists() and cache_saved is not None:
                    # Only an interrupted uncommitted cache can reach this path.
                    preserve_incomplete(cache_receipt)
                row = cache_scene(scene_path, sample['source_binding']['ss'], cache_path, encoder, resume=True)
                cached = validate_cached_sample(row, item, sample)
                atomic_json(receipt, {'identity_hash': identity, 'item': item, 'row': row,
                    'scene': binding(scene_path), 'source_files': cached['source_files'],
                    'camera_target_exact': True, 'completed_at_utc': utc()})
                del cached
        completed.append(row)
        if len(completed) % 2 == 0 or len(completed) == plan['occluded_count']:
            status = {'state': 'generating_occlusion', 'utc': utc(), 'identity_hash': identity,
                'completed': len(completed), 'total': plan['occluded_count'],
                'elapsed_seconds': time.monotonic() - started, 'last_uid': item['uid']}
            progress(output, status)
            print(json.dumps(status), flush=True)
    finalize(output, plan, identity, completed)
    print(json.dumps(read(output / 'status.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument('--manifest', default=str(SOURCE / 'train2000.json'))
    prepare_parser.add_argument('--dev-manifest', default=str(SOURCE / 'dev64.json'))
    prepare_parser.add_argument('--variants', type=int, choices=(0, 1, 2), default=2)
    prepare_parser.add_argument('--workers', type=int, default=2)
    prepare_parser.set_defaults(func=prepare)
    augment_parser = subparsers.add_parser('augment')
    augment_parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    augment_parser.add_argument('--device', default='cuda:0')
    augment_parser.add_argument('--cpu-threads', type=int, default=2)
    augment_parser.set_defaults(func=augment_data)
    args = parser.parse_args()
    if getattr(args, 'workers', 1) < 1 or getattr(args, 'cpu_threads', 1) < 1:
        raise ValueError('CPU worker/thread counts must be positive')
    try:
        args.func(args)
    except Exception:
        atomic_json(Path(args.output_dir) / 'failure.json', {'utc': utc(), 'command': args.command,
            'exception': traceback.format_exc(), 'policy': 'stop; no object or sample is silently omitted'})
        raise


if __name__ == '__main__':
    main()
