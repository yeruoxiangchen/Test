"""Independent scalar aggregation, training and artifact review (CPU only).

Does not call production summarize, resample surfaces, change frozen results or
load CUDA models. --write saves review receipts only, under the /data output.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
DATA = Path('/data/zjr/pose_aligned_layered_reconstruction')
for key, suffix in {
    'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
    'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
    'MPLCONFIGDIR': 'cache/matplotlib', 'NUMBA_CACHE_DIR': 'cache/numba',
    'XDG_CACHE_HOME': 'cache/xdg',
}.items():
    os.environ[key] = str(DATA / suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
sys.path.insert(0, str(ROOT))
c = importlib.import_module('pose_aligned_layered_reconstruction.experiments.hssd_depth_confidence_20261026.common')
import numpy as np
import torch
import trimesh

torch.set_num_threads(2)
ARMS = ('appearance_control', 'geometry_observed', 'geometry_gt_depth_pred_confidence',
        'geometry_pred_depth_unit_confidence', 'geometry_oracle_visible')
METRICS = ('chamfer_l1', 'fscore_0p01', 'fscore_0p02', 'normal_consistency', 'ss_iou',
           'ss_precision', 'ss_recall', 'visible_recall', 'occluded_recall', 'free_space_violation')
CONTRASTS = {
    'depth_at_predicted_confidence': (2, 1), 'depth_at_unit_confidence': (4, 3),
    'confidence_at_predicted_depth': (3, 1), 'confidence_at_gt_depth': (4, 2),
    'joint_gt_unit_minus_predicted': (4, 1),
    **{arm + '_minus_appearance': (i, 0) for i, arm in enumerate(ARMS) if i},
}
INTERACTION = 'depth_confidence_interaction'
COEFFICIENTS = np.array([[int(i == a) - int(i == b) for i in range(5)]
                         for a, b in CONTRASTS.values()] + [[0, 1, -1, -1, 1]], dtype=np.float64)
CONTRAST_NAMES = (*CONTRASTS, INTERACTION)
DIRECTION = np.array([-1 if m in ('chamfer_l1', 'free_space_violation') else 1 for m in METRICS])


class Audit:
    def __init__(self):
        self.checks = []; self.hashed = {}; self.skipped = {}
        self.exact_checks = 0; self.max_error = 0.

    def check(self, name, passed, **detail):
        if not bool(passed):
            raise AssertionError(name + ': ' + json.dumps(detail, ensure_ascii=False))
        self.checks.append({'check': name, 'passed': True, **detail})

    def equal(self, actual, expected, name):
        if isinstance(actual, dict):
            self.check(name + '/keys', set(actual) == set(expected))
            for key in actual: self.equal(actual[key], expected[key], name + '/' + key)
        elif isinstance(actual, (list, tuple)):
            self.check(name + '/length', len(actual) == len(expected))
            for i, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, name + '/' + str(i))
        elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
            error = abs(actual - expected)
            self.max_error = max(error, self.max_error); self.exact_checks += 1
            if not math.isfinite(error) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(name + ': numeric mismatch')
        else:
            self.exact_checks += 1
            if actual != expected: raise AssertionError(name + ': value mismatch')

    def bound(self, binding):
        key = (binding['path'], binding['sha256'])
        path = c.physical(binding['path'])
        if key in self.hashed or key in self.skipped: return path
        size = path.stat().st_size
        if size > 1024 * 1024:
            self.check('large_payload_on_data', str(path).startswith('/data/zjr/'), path=str(path))
        row = {'logical': binding['path'], 'physical': str(path), 'bytes': size, 'sha256': binding['sha256']}
        if size > 256 * 1024 * 1024:
            row['reason'] = 'Previously staged/bound large model; not byte-rehashed in this CPU review.'
            self.skipped[key] = row
        else:
            c.verify(binding); self.hashed[key] = row
        return path

    def load(self, binding): return c.read(self.bound(binding))

    def tree(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value: self.bound(value)
            if isinstance(value.get('checkpoint'), str) and 'sha256' in value:
                self.bound({'path': value['checkpoint'], 'sha256': value['sha256']})
            for k, v in value.items():
                if k not in ('path', 'sha256'): self.tree(v)
        elif isinstance(value, list):
            for v in value: self.tree(v)


def cell_dict(array): return {arm: dict(zip(METRICS, map(float, row))) for arm, row in zip(ARMS, array)}
def effect_dict(array): return {name: dict(zip(METRICS, map(float, row))) for name, row in zip(CONTRAST_NAMES, array)}
def effects(array): return np.einsum('ca,...am->...cm', COEFFICIENTS, array)


def win_dict(rows, signs=False):
    values = rows if signs else rows * DIRECTION
    labels = ('positive', 'zero', 'negative') if signs else ('better', 'equal', 'worse')
    return {m: dict(zip(labels, (int((values[:, j] > 1e-12).sum()),
                                int((np.abs(values[:, j]) <= 1e-12).sum()),
                                int((values[:, j] < -1e-12).sum()))), pairs=len(rows))
            for j, m in enumerate(METRICS)}


def raw_metrics(mesh, ss):
    surface, partitions, occupancy = mesh['surface'], mesh['partitioned'], ss['occupancy']
    return np.array([surface[m] for m in METRICS[:4]] +
                    [occupancy[m] for m in ('iou', 'precision', 'recall')] +
                    [partitions['visible']['recall_0p02'], partitions['externally_occluded']['recall_0p02'],
                     partitions['predicted_surface_free_space_violation_fraction']], dtype=np.float64)


def training_audit(a, p):
    policies = ('steps', 'seed', 'lr', 'warmup', 'betas', 'weight_decay', 'batch', 'p_uncond',
                'weights', 'amp', 'residual_channels', 'projection_bias', 'parent_checkpoint',
                'observation_variant', 'flow_loss', 'view_selection')
    step_fields = ('uid', 'index', 'original_index', 'epoch', 'offset', 'diffusion_time',
                   'noise_sha', 'learning_rate', 'object_group_exposures')
    init_fields = ('initial_residual_sha', 'trainable_parameter_count', 'trainable_parameter_names',
                   'bindings', 'fixed_parent')
    data_fields = ('uid', 'original_index', 'inputs', 'target_source', 'target_tensor_sha')
    runs = {}; histories = {}; completions = {}; sources = {}
    for arm in ARMS:
        source = ({k: c.binding(c.train_dir(arm) / name) for k, name in
                   [('run', 'run.json'), ('history', 'history.jsonl'), ('complete', 'complete.json')]}
                  if arm in c.ARMS else {k: p['reference_training'][arm][k] for k in ('run', 'history', 'complete')})
        run, complete = a.load(source['run']), a.load(source['complete'])
        history = [json.loads(line) for line in a.bound(source['history']).read_text().splitlines()]
        a.tree(run); a.tree(complete)
        a.check('complete_training/' + arm, [r['step'] for r in history] == list(range(1, 201))
                and complete['step'] == 200 and complete['contract'] == run['contract']
                and complete['data_identity'] == run['data_identity'] and run['trainable_parameter_count'] == 3164160
                and run['fixed_parent'] == 'complete old occluded appearance_control including its trained residual')
        a.check('finite_training/' + arm, all(math.isfinite(r[k]) for r in history
                for k in ('flow_loss', 'gradient_norm', 'learning_rate', 'diffusion_time')))
        a.check('equal_exposures/' + arm, dict(Counter(r['uid'] for r in history)) == {u: 50 for u in c.UIDS})
        if arm in c.ARMS:
            ckpt = torch.load(a.bound(complete['checkpoint']), map_location='cpu', weights_only=False)
            a.check('actual_checkpoint/' + arm, ckpt['step'] == 200 and ckpt['contract'] == run['contract']
                    and ckpt['data_identity'] == run['data_identity']
                    and c.tensor_hash(ckpt['residual']) == complete['final_residual_sha']
                    and ckpt['initial_residual_sha'] == run['initial_residual_sha']
                    and ckpt.get('frozen_parameter_versions_unchanged') is True)
            del ckpt
        runs[arm], histories[arm], completions[arm], sources[arm] = run, history, complete, source
    base = runs[ARMS[0]]
    for arm in ARMS:
        for key in policies: a.equal(runs[arm]['contract'][key], base['contract'][key], 'policy/' + arm + '/' + key)
        for key in init_fields: a.equal(runs[arm][key], base[key], 'initialization/' + arm + '/' + key)
        a.equal([{k: r[k] for k in data_fields} for r in runs[arm]['data_identity']],
                [{k: r[k] for k in data_fields} for r in base['data_identity']], 'inputs_targets/' + arm)
        for row, ref in zip(histories[arm], histories[ARMS[0]]):
            for key in step_fields: a.equal(row[key], ref[key], 'step/' + arm + '/' + str(row['step']) + '/' + key)
    return {'passed': True, 'sources': sources, 'steps_per_arm': 200, 'exposures_per_object': 50,
            'parameter_count': 3164160, 'same_initialization_parent_inputs_targets_noise_order_lr': True,
            'actual_new_checkpoints_verified': list(c.ARMS), 'paired_fields': step_fields,
            'runs': {arm: {'last_step_loss': histories[arm][-1]['flow_loss'],
                          'last50_mean_loss': float(np.mean([r['flow_loss'] for r in histories[arm][-50:]])),
                          'final_residual_sha': completions[arm]['final_residual_sha']} for arm in ARMS}}


def tensor_receipt(value):
    v = value.detach().contiguous().cpu()
    return {'shape': list(v.shape), 'dtype': str(v.dtype),
            'sha256': hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def review():
    a = Audit(); wrapper = c.plan(); p = wrapper['plan']; out = c.OUT
    summary = c.read(out / 'summary.json'); runtime = c.read(out / 'evaluation_identity.json'); identity = c.digest(runtime)
    a.check('identity', summary['identity'] == identity and runtime['plan_identity'] == wrapper['identity'])
    a.check('counts', (summary['new_meshes'], summary['new_ss_samples'], summary['original_references'],
                     summary['logical_cases'], summary['factorial_cases']) == (24, 24, 36, 60, 48))
    a.check('cases', len(p['cases']) == 60 and {(r['uid'], r['seed'], r['arm']) for r in p['cases']} ==
            {(u, s, arm) for u in c.UIDS for s in c.SEEDS for arm in ARMS})
    a.tree(runtime); a.tree(summary['smoke'])
    # The plan also retains provenance-only raw scene/asset paths on home.
    # Inspect the staged inputs actually consumed by this experiment, without
    # reopening those original assets or recursively reauditing old pipelines.
    for key in ('previous_evaluation_identity', 'previous_summary', 'previous_plan', 'previous_review',
                'reference_visual_audits', 'storage_manifest', 'parent_checkpoint', 'parent_training_record',
                'slat_binding', 'predicted_depth', 'stripped_observations', 'reference_evidence'):
        a.tree(p[key])
    a.check('frozen_source_inventory', runtime['sources'] == [c.binding(f) for f in
            sorted(c.HERE.glob('*.py')) + [c.HERE / 'run.sh', c.HERE / 'PROTOCOL.md']])
    smoke = c.read(out / 'smoke.json')
    a.check('smoke', smoke['identity'] == identity and smoke['arms'] == list(c.ARMS) and all(smoke[k] is True for k in
            ('passed', 'parent_ss_mesh_replay_exact', 'zero_initialization_exact', 'matched_trainable_capacity',
             'two_step_training_passed', 'shared_noise_order_exact', 'diagonal_evidence_replay_exact')))
    for key in ('initial_check', 'training_check', 'replay'):
        receipt = a.load(smoke[key]); a.tree(receipt); a.check('smoke/' + key, receipt['passed'] is True)
    diagonal = a.load(smoke['diagonal_check']); a.tree(diagonal)
    a.check('diagonal_count', diagonal['passed'] is True and len(diagonal['checks']) == 8
            and all(r['all_tensors_exact'] for r in diagonal['checks']))
    evidence = c.read(out / 'evidence/complete.json'); a.tree(evidence)
    a.check('new_evidence_complete', evidence['success'] is True and len(evidence['objects']) == 8)
    for binding in evidence['objects']:
        receipt = a.load(binding); a.tree(receipt)
    for binding in p['stripped_observations'].values():
        receipt = a.load(binding); a.tree(receipt)
        with np.load(a.bound(receipt['npz'])) as observations:
            a.check('visible_first_surface_allowlist', set(observations.files) == {'depth_z_world', 'valid'})
    initial_count = a.exact_checks
    training = training_audit(a, p)
    training_checks = a.exact_checks - initial_count
    print(json.dumps({'stage': 'training_audited', 'checks': training_checks}), flush=True)
    source_inputs = {}
    base = a.load(training['sources']['appearance_control']['run'])
    for obj in p['objects']:
        row = next(r for r in base['data_identity'] if r['uid'] == obj['uid'])
        sample = torch.load(a.bound(row['inputs']['sample']), map_location='cpu', weights_only=True)
        with np.load(a.bound(obj['ss'])) as target:
            a.check('actual_target/' + obj['uid'], np.array_equal(sample['target'].numpy(), target['target'])
                    and np.array_equal(sample['target_coords'].numpy(), target['official_coords']))
        a.check('target_frame/' + obj['uid'], sample['canonical_from_world'] == obj['canonical_from_world']
                and sample['grid_transform'] == 'identity'
                and c.tensor_hash({'target': sample['target']}) == row['target_tensor_sha'])
        source_inputs[obj['uid']] = row['inputs']
    array = np.empty((4, 3, 5, 10), dtype=np.float64); coordinate_rows = []; sources = []; fingerprints = {}
    indexed = {(r['uid'], r['seed'], r['arm']): r for r in summary['records']}
    a.check('summary_unique_records', len(indexed) == len(summary['records']) == 60)
    for count, case in enumerate(p['cases'], 1):
        key = case['uid'], case['seed'], case['arm']; saved = indexed[key]
        obj = next(o for o in p['objects'] if o['uid'] == case['uid'])
        if case['is_reference']: mb, sb = case['reference_mesh'], case['reference_ss']
        else: mb, sb = c.binding(c.case_dir(case) / 'mesh.json'), c.binding(c.case_dir(case) / 'ss.json')
        mesh, ss = a.load(mb), a.load(sb); a.tree(mesh); a.tree(ss); sources.extend((mb, sb))
        expected = case['expected_identity'] if case['is_reference'] else identity
        a.check('case_binding/' + str(key), mesh['success'] and ss['success'] and mesh['identity'] == ss['identity'] == expected
                and saved['mesh_record'] == mb and saved['ss_record'] == sb and saved['mesh'] == mesh and mesh['ss'] == sb
                and mesh['object_uid'] == ss['object_uid'] == case['uid'] and mesh['seed'] == ss['seed'] == case['seed']
                and mesh['arm'] == ss['arm'] == case['arm'])
        condition = a.load(case['condition_record'])
        a.check('input_and_frame/' + str(key), mesh['inputs'] == ss['inputs'] == condition['inputs'] == source_inputs[case['uid']]
                and mesh['canonical_from_world'] == obj['canonical_from_world'] and mesh['gt'] == obj['mesh']
                and mesh['fitted_transform'] is False and mesh['slat_binding'] == p['slat_binding']
                and mesh['calls'] == ss['calls'] == c.CALLS
                and tuple(mesh['partitioned'][k]['gt_samples'] for k in ('visible', 'externally_occluded', 'other_unobserved'))
                == c.EXPECTED_PARTITIONS[case['uid']])
        vals = raw_metrics(mesh, ss); a.check('finite_metrics/' + str(key), np.isfinite(vals).all())
        array[c.UIDS.index(case['uid']), c.SEEDS.index(case['seed']), ARMS.index(case['arm'])] = vals
        a.equal(dict(zip(METRICS, vals.tolist())), saved['metrics'], 'raw_metrics/' + str(key))
        with np.load(a.bound(ss['coords'])) as payload: coords = payload['coords']
        a.check('support/' + str(key), coords.dtype == np.int32 and coords.shape == (ss['occupancy']['predicted_voxels'], 4)
                and len(coords) > 0 and np.all(coords[:, 0] == 0) and np.all((coords[:, 1:] >= 0) & (coords[:, 1:] < 64))
                and np.array_equal(np.unique(coords, axis=0), coords))
        receipt = a.load(mesh['sampled_slat_receipt']); a.tree(receipt)
        latent = torch.load(a.bound(receipt['latent']), map_location='cpu', weights_only=True)
        a.check('latent/' + str(key), np.array_equal(latent['coords'].numpy(), coords)
                and latent['feats'].shape == (len(coords), 8) and torch.isfinite(latent['feats']).all()
                and receipt['contract']['inputs'] == ss['inputs'] and receipt['contract']['slat_binding'] == p['slat_binding']
                and receipt['calls'] == c.CALLS
                and receipt['contract']['master_seed'] == case['seed'] * 2000003 + obj['original_index'] * 2017 + 7919
                and tensor_receipt(latent['coords']) == receipt['contract']['initial_coords'])
        canonical = trimesh.load(a.bound(mesh['mesh']), process=False, force='mesh')
        a.check('mesh_structure/' + str(key), np.isfinite(canonical.vertices).all()
                and len(canonical.vertices) == mesh['structure']['vertex_count']
                and len(canonical.faces) == mesh['structure']['face_count'])
        fingerprints[key] = {'coords': hashlib.sha256(coords.tobytes()).hexdigest(),
                             'latent': tensor_receipt(latent['feats'])['sha256'],
                             'mesh': hashlib.sha256(canonical.vertices.tobytes() + canonical.faces.tobytes()).hexdigest()}
        if not case['is_reference']:
            world = trimesh.load(a.bound(mesh['world_mesh']), process=False, force='mesh')
            cfw = np.asarray(obj['canonical_from_world'], dtype=np.float64); wfc = np.linalg.inv(cfw)
            world_error = float(np.max(np.abs(world.vertices - (canonical.vertices @ wfc[:3, :3].T + wfc[:3, 3]))))
            back_error = float(np.max(np.abs(world.vertices @ cfw[:3, :3].T + cfw[:3, 3] - canonical.vertices)))
            a.check('full_coordinate_mapping/' + str(key), np.array_equal(world.faces, canonical.faces)
                    and world_error <= 1e-6 and back_error <= 1e-6
                    and abs(world_error - mesh['world_export_max_abs_error']) <= 1e-12)
            coordinate_rows.append({'arm': case['arm'], 'uid': case['uid'], 'seed': case['seed'],
                                    'vertices': len(canonical.vertices), 'faces': len(canonical.faces),
                                    'world_max_abs_error_m': world_error, 'canonical_roundtrip_error': back_error})
        if count % 15 == 0: print(json.dumps({'stage': 'case_review', 'complete': count, 'total': 60}), flush=True)
    for arm in c.ARMS:
        for uid in c.UIDS:
            for field in ('coords', 'latent', 'mesh'):
                a.check('sampling_seed_independence/' + arm + '/' + uid + '/' + field,
                        len({fingerprints[uid, s, arm][field] for s in c.SEEDS}) == 3)
            for seed in c.SEEDS:
                for other in ARMS:
                    if other != arm:
                        a.check('distinct_new_results/' + str((uid, seed, arm, other)),
                                all(fingerprints[uid, seed, arm][f] != fingerprints[uid, seed, other][f]
                                    for f in ('coords', 'latent', 'mesh')))
    statistics = summary['statistics']; per_pair_effect = effects(array); objects = array.mean(axis=1)
    object_effect = effects(objects); macro = objects.mean(axis=0); macro_effect = effects(macro)
    for i, uid in enumerate(c.UIDS):
        for j, seed in enumerate(c.SEEDS):
            a.equal({'cells': cell_dict(array[i, j]), 'contrasts': effect_dict(per_pair_effect[i, j])},
                    statistics['per_seed'][uid][str(seed)], 'paired/' + uid + '/' + str(seed))
        row = statistics['per_object'][uid]
        a.equal(cell_dict(objects[i]), row['cells'], 'object_cells/' + uid)
        a.equal(effect_dict(object_effect[i]), row['contrasts'], 'object_effects/' + uid)
        for k, name in enumerate(CONTRAST_NAMES):
            a.equal({m: [float(per_pair_effect[i, :, k, n].min()), float(per_pair_effect[i, :, k, n].max())]
                     for n, m in enumerate(METRICS)}, row['delta_seed_range'][name], 'seed_range/' + uid + '/' + name)
            if name != INTERACTION:
                a.equal(win_dict(per_pair_effect[i, :, k]), row['seed_win_counts'][name], 'seed_wins/' + uid + '/' + name)
        a.equal(win_dict(per_pair_effect[i, :, -1], signs=True), row['interaction_seed_sign_counts'], 'interaction_signs/' + uid)
    a.equal(cell_dict(macro), statistics['macro']['cells'], 'macro_cells')
    a.equal(effect_dict(macro_effect), statistics['macro']['contrasts'], 'macro_effects')
    a.equal(win_dict(object_effect[:, -1], signs=True), statistics['macro']['interaction_object_sign_counts'], 'macro_interaction')
    for k, name in enumerate(CONTRASTS):
        a.equal(win_dict(per_pair_effect[:, :, k].reshape(12, 10)), statistics['pair_win_counts'][name], 'pair_wins/' + name)
        a.equal(win_dict(object_effect[:, k]), statistics['macro']['object_win_counts'][name], 'object_wins/' + name)
    visual = a.load(summary['visual_audit']); a.tree(visual)
    a.check('visual_counts', len(visual['figures']) == 12 and len(visual['render_records']) == 64
            and len(visual['worlds']) == 15 and visual['new_mesh_render_count'] == 24
            and not visual['clipping'] and visual['archived_gt_render_count'] == 4 and visual['archived_method_render_count'] == 36)
    for binding in visual['worlds']:
        a.check('world_four_objects', binding['objects'] == 4 and len(binding['sources']) == 4)
    for record in visual['render_records']:
        value = a.load(record['record']); a.tree(value)
        a.check('native_render_transform', np.array_equal(np.asarray(value['matrix_original_to_rendered']), np.eye(4)) and len(value['views']) == 3)
        for view in value['views']:
            a.bound({'path': view['png'], 'sha256': view['png_sha256']})
            a.check('render_unclipped', view['vertices_outside_xy'] == view['vertices_outside_depth'] == 0
                    and view['foreground_touches_image_border'] is False)
    a.check('cpu_only', not torch.cuda.is_initialized())
    analysis = {'time': c.now(), 'identity': identity, 'passed': True,
                'scope': 'Independent reaggregation of original scalar records, not new surface sampling or GPU generation.',
                'macro': {'cells': cell_dict(macro), 'contrasts': effect_dict(macro_effect)},
                'per_object': {uid: {'cells': cell_dict(objects[i]), 'contrasts': effect_dict(object_effect[i])} for i, uid in enumerate(c.UIDS)},
                'pair_win_counts': statistics['pair_win_counts'], 'object_win_counts': statistics['macro']['object_win_counts'],
                'leave_one_object_out': {uid: {'cells': cell_dict(np.delete(objects, i, axis=0).mean(axis=0)),
                                              'contrasts': effect_dict(effects(np.delete(objects, i, axis=0).mean(axis=0)))} for i, uid in enumerate(c.UIDS)},
                'leave_one_sampling_seed_out': {str(seed): {'cells': cell_dict(np.delete(array, j, axis=1).mean(axis=(0, 1))),
                                                           'contrasts': effect_dict(effects(np.delete(array, j, axis=1).mean(axis=(0, 1))))} for j, seed in enumerate(c.SEEDS)},
                'training_fairness': training, 'source_record_bindings': sources,
                'limitations': ['Four previously trained objects, one scene; no held-out evidence.',
                                'Three seeds are sampling repeats, not independently trained models.',
                                'GT depth also supplies its own validity; this is a depth-source comparison.',
                                'Unit confidence changes filtering and attenuation together.',
                                'Known cameras, masks and canonical frames remain assisted.',
                                'Visual observations are agent inspection; no user review is inferred.']}
    verification = {'time': c.now(), 'passed': True, 'identity': identity, 'plan_identity': wrapper['identity'],
                    'cases': 60, 'new_cases': 24, 'references': 36, 'checks_passed': len(a.checks),
                    'scalar_or_exact_value_checks': a.exact_checks, 'training_field_checks': training_checks,
                    'max_abs_aggregation_error': a.max_error, 'files_hashed': len(a.hashed),
                    'large_bindings_not_rehashed': list(a.skipped.values()),
                    'coordinate_rows': coordinate_rows, 'new_vertices_checked': sum(r['vertices'] for r in coordinate_rows),
                    'max_world_error_m': max(r['world_max_abs_error_m'] for r in coordinate_rows),
                    'max_roundtrip_error': max(r['canonical_roundtrip_error'] for r in coordinate_rows),
                    'render_records_checked': 64, 'render_views_checked': 192, 'figures_checked': 12,
                    'world_exports_bound': 15, 'no_fitted_transform': True, 'cpu_only': True,
                    'checks': a.checks, 'file_bindings_verified': list(a.hashed.values()),
                    'review_source': c.binding(Path(__file__)),
                    'provenance_only_raw_assets_reopened': False}
    return analysis, verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); analysis, verification = review()
    if args.write:
        c.write(c.OUT / 'review_20261026/analysis.json', analysis)
        c.write(c.OUT / 'review_20261026/verification.json', verification)
    print(json.dumps({k: v for k, v in verification.items() if k not in
                      ('checks', 'file_bindings_verified', 'coordinate_rows', 'large_bindings_not_rehashed')}, ensure_ascii=False, indent=2))
