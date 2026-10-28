"""Complete frozen SLat samples with independently bound support and observation."""
import argparse
import hashlib
import os
import numpy as np
import torch

from .common import *
from .prepare import partition_counts
from ...frozen_slat import (load_frozen_slat30k, master_noise, sparse_noise,
                           sample_mesh_latent, configure_determinism, bootstrap_external_paths)
from ...mesh_metrics import surface_metrics, structure_metrics
from ..hssd_input_adaptation_20261025.evaluate import load_input
from ..hssd_shared_scene_20261025.evaluate import partitioned_metrics
from ..hssd_input_adaptation_int32_20261025.tiled_native import LargeInputNative
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh


def tensor_binding(value):
    value = value.detach().contiguous().cpu()
    return {'shape':list(value.shape), 'dtype':str(value.dtype),
            'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def load_support(case):
    row = read(verify(case['support_record']))
    if (row['identity'] != plan()['plan']['base_identity'] or row['object_uid'] != case['uid']
            or row['seed'] != case['seed'] or row['variant'] != variant(case['support_level'])
            or row['arm'] != ARM or not row['success']):
        raise ValueError('Support is not the frozen requested SS result')
    with np.load(verify(row['coords'])) as data:
        coords = data['coords']
    return row, coords


def load_condition(case, cache):
    obj = object_row(case)
    source, sample, inputs = load_input(obj, variant(case['condition_level']), cache)
    reference = read(verify(case['condition_record']))
    if (reference['identity'] != plan()['plan']['base_identity'] or reference['object_uid'] != case['uid']
            or reference['seed'] != case['seed'] or reference['variant'] != variant(case['condition_level'])
            or reference['arm'] != ARM or reference['inputs'] != inputs):
        raise ValueError('Condition is not the requested complete observation package')
    return obj, sample, inputs


@torch.no_grad()
def noise_audit(device, ident):
    rows = []
    for obj in plan()['plan']['objects']:
        for seed in SEEDS:
            cases = [next(c for c in logical_cases() if case_key(c) == (obj['uid'],seed,level,level)) for level in LEVELS]
            supports = [load_support(c)[1] for c in cases]
            field = master_noise(seed, obj['original_index'], device)
            if not torch.equal(field, master_noise(seed, obj['original_index'], device)):
                raise ValueError('Master noise is not repeatable')
            tensors = [sparse_noise(coords, field) for coords in supports]
            codes = [np.ravel_multi_index(coords[:,1:].T, (64,64,64)) for coords in supports]
            common, ai, bi = np.intersect1d(codes[0],codes[1],return_indices=True)
            if len(common) and not torch.equal(tensors[0].feats[torch.as_tensor(ai,device=device)],
                                               tensors[1].feats[torch.as_tensor(bi,device=device)]):
                raise ValueError('Noise differs at a common canonical coordinate')
            rows.append({'uid':obj['uid'], 'seed':seed, 'original_index':obj['original_index'],
                'master_seed':seed*2000003+obj['original_index']*2017+7919,
                'master':tensor_binding(field), 'common_coordinates':len(common),
                'common_noise_equal':True, 'comparison_vacuous':not bool(len(common)),
                'initials':{level:{'coords':tensor_binding(t.coords), 'features':tensor_binding(t.feats)}
                            for level,t in zip(LEVELS,tensors)}})
            del field, tensors
    write(OUT/'noise_audit.json', {'time':now(), 'identity':ident, 'passed':True, 'rows':rows,
        'contract':'Same original seed/index and canonical-coordinate noise; different support need not have equal length or overlap'})
    event('coordinate_noise_audit_complete', paired_supports=len(rows))


@torch.no_grad()
def sample_or_resume(dest, contract, runtime, sample, initial):
    from trellis.modules import sparse as sp
    path = dest/'sampled_slat_receipt.json'
    if path.exists():
        row = read(path)
        if row['contract'] != contract:
            raise ValueError('Refuse a cached latent with different support or condition')
        saved = torch.load(verify(row['latent']), map_location='cpu', weights_only=True)
        latent = sp.SparseTensor(coords=saved['coords'].to(initial.device), feats=saved['feats'].to(initial.device))
        calls = row['calls']
        event('reuse_sampled_slat', case=case_key(contract['case']))
    else:
        event('sample_slat', case=case_key(contract['case']), support=len(initial.coords))
        latent, calls = sample_mesh_latent(runtime, sample, initial)
        temporary = dest/f'sampled_slat.{os.getpid()}.tmp'
        torch.save({'coords':latent.coords.detach().cpu(),'feats':latent.feats.detach().cpu()}, temporary)
        temporary.replace(dest/'sampled_slat.pt')
        write(path, {'time':now(), 'contract':contract, 'calls':calls, 'latent':binding(dest/'sampled_slat.pt')})
        event('saved_slat_before_decode', case=case_key(contract['case']), support=len(latent.coords))
    if (calls != CALLS or not torch.equal(latent.coords, initial.coords)
            or latent.feats.shape != initial.feats.shape or not torch.isfinite(latent.feats).all()):
        raise ValueError('Incomplete/nonfinite/wrong-coordinate SLat')
    return latent, calls, binding(path)


def verify_diagonal(record, case):
    import trimesh
    old = read(verify(case['diagonal_mesh']))
    for field in ['mesh', 'world_mesh']:
        actual = trimesh.load(verify(record[field]), process=False, force='mesh')
        archived = trimesh.load(verify(old[field]), process=False, force='mesh')
        if not np.array_equal(actual.vertices, archived.vertices) or not np.array_equal(actual.faces, archived.faces):
            raise ValueError('Diagonal replay differs: '+field)
    for field in ['surface', 'structure', 'partitioned', 'slat_binding', 'calls',
                  'canonical_from_world', 'fitted_transform', 'gt_partition_scope', 'inputs', 'ss']:
        if record[field] != old[field]:
            raise ValueError('Diagonal replay field differs: '+field)


@torch.no_grad()
def run_cases(cases, ident):
    import trimesh
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    install_memory_efficient_mesh()
    cache, runtime = {}, None
    shared = read(verify(plan()['plan']['shared_plan']))['plan']
    records = []
    with LargeInputNative() as policy:
        for case in cases:
            if case not in logical_cases():
                raise ValueError('Case not in the frozen plan')
            obj, sample, inputs = load_condition(case, cache)
            support, coords = load_support(case)
            dest = replay_dir(case) if case['is_reference'] else case_dir(case)
            dest.mkdir(parents=True, exist_ok=True)
            path = dest/'mesh.json'
            if path.exists():
                record = validate_saved_result(read(path), case, ident, inputs)
                if case['is_reference']:
                    verify_diagonal(record, case)
                records.append(binding(path)); continue
            if runtime is None:
                event('load_frozen_slat')
                runtime = load_frozen_slat30k(device, evaluation_uids=plan()['plan']['evaluation_uid_context'])
                if runtime['binding'] != plan()['plan']['slat_binding']:
                    raise ValueError('Frozen SLat differs from archived diagonal')
            initial = sparse_noise(coords, master_noise(case['seed'], obj['original_index'], device))
            support_source = {'record':case['support_record'], 'coords':support['coords'], 'inputs':support['inputs']}
            condition_source = {'record':case['condition_record'], 'inputs':inputs}
            contract = {'identity':ident, 'case':case, 'support_source':support_source,
                'condition_source':condition_source, 'slat_binding':runtime['binding'],
                'original_index':obj['original_index'], 'master_seed':case['seed']*2000003+obj['original_index']*2017+7919,
                'initial_coords':tensor_binding(initial.coords), 'initial_noise':tensor_binding(initial.feats)}
            tiled_start = len(policy.tiled_calls)
            latent, calls, receipt = sample_or_resume(dest, contract, runtime, sample, initial)
            event('decode_mesh', case=case_key(case))
            mesh = runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError('Invalid mesh; preserve this case as a failure')
            mesh.export(dest/'mesh_canonical.ply')
            mesh = trimesh.load(dest/'mesh_canonical.ply', process=False, force='mesh')
            target = trimesh.load(verify(obj['mesh']), process=False, force='mesh')
            world = mesh.copy(); world.apply_transform(np.linalg.inv(obj['canonical_from_world']))
            world.export(dest/'mesh_world.ply')
            world_saved = trimesh.load(dest/'mesh_world.ply', process=False, force='mesh')
            error = float(np.max(np.abs(world_saved.vertices-world.vertices)))
            if error > 1e-6 or not np.array_equal(world_saved.faces, mesh.faces):
                raise ValueError('World/canonical export mismatch')
            part = partitioned_metrics(mesh, target, obj, shared)
            old_diag = read(base.case_dir(variant(case['support_level']), ARM, obj['uid'], case['seed'])/'mesh.json')
            if partition_counts({'partitioned':part}) != partition_counts(old_diag):
                raise ValueError('GT partition membership changed')
            record = {'identity':ident, 'case':case, 'object_uid':case['uid'], 'seed':case['seed'], 'arm':ARM,
                'support_level':case['support_level'], 'condition_level':case['condition_level'],
                'support_source':support_source, 'condition_source':condition_source,
                'inputs':inputs, 'condition_inputs':inputs, 'ss':case['support_record'],
                'success':True, 'fitted_transform':False, 'canonical_from_world':obj['canonical_from_world'],
                'mesh':binding(dest/'mesh_canonical.ply'), 'world_mesh':binding(dest/'mesh_world.ply'), 'gt':obj['mesh'],
                'surface':surface_metrics(mesh,target), 'structure':structure_metrics(mesh), 'partitioned':part,
                'slat_binding':runtime['binding'], 'calls':calls, 'kernel_policy':policy.manifest(),
                'native_input_row_tiling':policy.tiled_calls[tiled_start:],
                'sampled_slat_receipt':receipt, 'world_export_max_abs_error':error,
                'runtime_amendment':plan()['plan']['runtime_amendment'],
                'gt_partition_scope':'fixed original four shared views for every variant'}
            if case['is_reference']:
                verify_diagonal(record, case)
                record['diagonal_exact'] = True
            validate_saved_result(record, case, ident, inputs)
            write(path, record)
            records.append(binding(path))
            event('mesh_case_complete', case=case_key(case), reference_replay=case['is_reference'])
            del initial, latent, mesh, target, world, world_saved
    return records


@torch.no_grad()
def main(action):
    configure_determinism(); bootstrap_external_paths()
    ident = freeze_runtime()
    preflight = read(OUT/'preflight.json')
    if not preflight['passed'] or preflight['identity'] != ident:
        raise ValueError('CPU preflight required')
    if action == 'smoke':
        noise_audit(torch.device('cuda:0'), ident)
        records = run_cases(smoke_cases(), ident)
        write(OUT/'smoke.json', {'time':now(), 'identity':ident, 'passed':True,
            'diagonal_exact':True, 'cross_complete':True, 'coordinate_noise_exact':True,
            'noise_audit':binding(OUT/'noise_audit.json'), 'records':records,
            'new_cross_meshes':1, 'diagonal_replays':1})
        event('smoke_complete', diagonal_exact=True, completed_new_meshes=1)
    else:
        check_smoke()
        records = run_cases(new_jobs(), ident)
        write(OUT/'mesh_complete.json', {'time':now(), 'identity':ident, 'new_meshes':12, 'records':records})
        event('mesh_complete', new_meshes=12, logical_cases=24)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['smoke','mesh'])
    main(p.parse_args().action)
