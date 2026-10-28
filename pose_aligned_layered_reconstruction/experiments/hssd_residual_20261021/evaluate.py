"""Full native-frame mesh evaluation against the identical trained appearance base."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ...v2.dataset import SceneDataset, select_views
from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent, SAMPLER_PARAMS
from ...pretrained import load_trellis_model
from ...mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import ARMS, FORMAT, OUT, PILOT, binding, policy, verify, verify_old_sources, write, now
from .model import build, ResidualCFG, observations


def copy_verified(source, destination):
    source = verify(source)
    if destination.exists():
        if binding(destination)['sha256'] != binding(source)['sha256']:
            raise ValueError('Changed copied mesh: ' + str(destination))
    else:
        shutil.copyfile(source, destination)


def prepare_baseline(domain, arm, root, contract):
    """Reuse all32 original no_layer meshes only after exact protocol checks."""
    verify_old_sources()
    old = PILOT / 'evaluations' / f'{domain}_ss_no_layer'
    prior = json.loads((old / 'candidate_complete.json').read_text())
    expected = {'manifest': contract['manifest'], 'checkpoint': contract['appearance_base'],
                'stage': 'ss', 'weights': 'raw', 'seeds': [42, 43], 'sampler': SAMPLER_PARAMS, 'max_samples': 16}
    for key, value in expected.items():
        if prior[key] != value:
            raise ValueError('Original appearance baseline protocol mismatch: ' + key)
    dataset = SceneDataset(verify(contract['manifest']), require_labels=False)
    bindings = []
    for index, row in enumerate(dataset.rows):
        folder = root / f'{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for seed in [42, 43]:
            path = old / f'{index:03d}' / f'{seed}_candidate.json'
            record = json.loads(path.read_text())
            if record['uid'] != row['uid'] or record['seed'] != seed or not record['success']:
                raise ValueError('Original baseline has missing/unmatched/failed cases')
            copy_verified(record['mesh'], folder / f'{seed}_parent.ply')
            copy_verified(record['gt'], folder / 'gt.ply')
            current = {**record, 'mode': 'parent', 'mesh': binding(folder / f'{seed}_parent.ply'),
                       'gt': binding(folder / 'gt.ply'), 'original_record': binding(path)}
            write(folder / f'{seed}_parent.json', current)
            bindings.append(binding(path))
    write(root / 'parent_complete.json', {'contract': contract, 'original_complete': binding(old / 'candidate_complete.json'),
        'source_records': bindings, 'scope': 'identical frozen trained no_layer base; original native-frame meshes'})


class InvalidMesh(ValueError):
    pass


@torch.no_grad()
def evaluate(domain, arm):
    import trimesh
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    contract = policy(domain, arm)
    checkpoint_file = OUT / 'train' / f'{domain}_{arm}' / 'last.pt'
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    if checkpoint['format'] != FORMAT or checkpoint['contract'] != contract or checkpoint['step'] != contract['steps']:
        raise ValueError('Evaluation requires completed matching residual training')
    root = OUT / 'evaluations' / f'{domain}_{arm}'
    root.mkdir(parents=True, exist_ok=True)
    run_identity = {'contract': contract, 'checkpoint': binding(checkpoint_file), 'seeds': [42, 43],
        'sampler': SAMPLER_PARAMS, 'max_samples': 16, 'gt_frame': 'native; no fitted rotation/translation/scale',
        'privileged_gt_geometry': arm == 'oracle', 'weights': 'raw'}
    identity_file = root / 'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != run_identity:
        raise ValueError('Changed evaluation identity')
    write(identity_file, run_identity)
    prepare_baseline(domain, arm, root, contract)
    install_memory_efficient_mesh()
    sampler, model, saved, bindings = build(contract, device)
    if checkpoint['bindings'] != bindings:
        raise ValueError('Frozen base binding changed')
    model.residual.load_state_dict(checkpoint['residual'], strict=True)
    model.eval().requires_grad_(False)
    ss_decoder, _ = load_trellis_model(saved['pretrained'], 'ss_dec_conv3d_16l8_fp16')
    ss_decoder = ss_decoder.to(device).eval()
    runtime = load_frozen_slat30k(device)
    runtime['model'].cpu()
    dataset = SceneDataset(verify(contract['manifest']), require_labels=arm == 'oracle')
    for index in range(16):
        source = dataset[index]
        model_input = source if arm == 'oracle' else observations(source)
        sample = select_views(model_input, torch.arange(len(source['intrinsics'])))
        folder = root / f'{index:03d}'
        gt_path = folder / 'gt.ply'
        # Reused GT must match the actual mesh bound by this sample.
        target = trimesh.load(verify(source['target_mesh_binding']), force='mesh', process=False)
        reused = trimesh.load(gt_path, force='mesh', process=False)
        if not (np.array_equal(target.vertices, reused.vertices) and np.array_equal(target.faces, reused.faces)):
            # PLY serialization may round floating coordinates; compare to the
            # exact PLY-roundtripped source, never fit a transform.
            import io
            roundtrip = trimesh.load(io.BytesIO(target.export(file_type='ply')), file_type='ply', process=False)
            if not (np.array_equal(roundtrip.vertices, reused.vertices) and np.array_equal(roundtrip.faces, reused.faces)):
                raise ValueError('Native-frame GT mismatch')
        target = reused
        model.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prepared = model.prepare(sample, device)
        for seed in [42, 43]:
            record_file = folder / f'{seed}_candidate.json'
            if record_file.exists():
                row = json.loads(record_file.read_text())
                verify(row['gt'])
                if row['success']:
                    verify(row['mesh'])
                continue
            base_record = {'uid': source['uid'], 'object_uid': source['object_uid'], 'seed': seed,
                'mode': 'candidate', 'gt': binding(gt_path), 'privileged_gt_geometry': arm == 'oracle',
                'support_kind': 'predicted_ss', 'gt_kind': source['label_geometry_kind']}
            try:
                model.to(device)
                positive = sample['stock_condition'].to(device)
                wrapper = ResidualCFG(model, positive, prepared)
                generator = torch.Generator(device=device).manual_seed(seed * 1000003 + index * 1009)
                noise = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = sampler.sample(wrapper, noise, cond=positive, neg_cond=torch.zeros_like(positive),
                        steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
                    logits = ss_decoder(latent.to(next(ss_decoder.parameters()).dtype))
                if not torch.isfinite(logits).all() or not torch.isfinite(latent).all():
                    raise InvalidMesh('nonfinite SS output')
                coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)
                np.savez_compressed(folder / f'{seed}_candidate_ss.npz', coords=coords)
                base_record.update(active_voxels=len(coords), positive_calls=wrapper.positive_calls, negative_calls=wrapper.negative_calls)
                del latent, logits, wrapper, noise
                model.cpu()
                torch.cuda.empty_cache()
                if not len(coords):
                    raise InvalidMesh('empty SS support')
                runtime['model'].to(device)
                initial = sparse_noise(coords, master_noise(seed, index, device))
                z, _ = sample_mesh_latent(runtime, sample, initial)
                runtime['model'].cpu()
                torch.cuda.empty_cache()
                mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
                if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                    raise InvalidMesh('empty or nonfinite decoded mesh')
                mesh_path = folder / f'{seed}_candidate.ply'
                mesh.export(mesh_path)
                row = {**base_record, 'success': True, 'surface': surface_metrics(mesh, target),
                       'structure': structure_metrics(mesh), 'mesh': binding(mesh_path)}
                del mesh, z, initial
            except InvalidMesh as exc:
                row = {**base_record, 'success': False, 'failure': str(exc)}
            write(record_file, row)
            print(json.dumps({'time': now(), 'index': index, 'seed': seed, 'arm': arm,
                              'success': row['success'], 'surface': row.get('surface')}), flush=True)
            torch.cuda.empty_cache()
    write(root / 'candidate_complete.json', run_identity)


def summarize(domain, arm):
    root = OUT / 'evaluations' / f'{domain}_{arm}'
    candidate = json.loads((root / 'candidate_complete.json').read_text())
    parent = json.loads((root / 'parent_complete.json').read_text())
    if candidate['contract'] != parent['contract'] or candidate['contract'] != policy(domain, arm):
        raise ValueError('Summary identity mismatch')
    verify(candidate['checkpoint'])
    dataset = SceneDataset(verify(candidate['contract']['manifest']), require_labels=False)
    records = []
    for index, sample in enumerate(dataset.rows):
        for seed in [42, 43]:
            for mode in ['parent', 'candidate']:
                row = json.loads((root / f'{index:03d}' / f'{seed}_{mode}.json').read_text())
                if row['uid'] != sample['uid'] or row['mode'] != mode or row['seed'] != seed:
                    raise ValueError('Unpaired evaluation record')
                verify(row['gt'])
                if row['success']:
                    verify(row['mesh'])
                records.append(row)
    failures = [r for r in records if not r['success']]
    if failures:
        result = {'all_meshes_successful': False, 'failed_cases': failures,
                  'fit_advantage_established': False, 'note': 'No success-only metric aggregate'}
    else:
        result = aggregate_mesh_records(records, [r['uid'] for r in dataset.rows], seeds=[42, 43])
        cd = result['paired']['chamfer_l1']
        result['fit_advantage_established'] = cd['mean_improvement'] > 0 and cd['win_objects'] >= 8
        from ..hssd_pilot_20261020.evaluate import galleries
        result['galleries'] = galleries(root, 16, [42, 43])
    result.update(contract=candidate['contract'], privileged_gt_geometry=arm == 'oracle',
                  expansion_authorized=False, scope='training-set fit diagnostic; compare predicted vs appearance_control as well')
    write(root / 'summary.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['clean', 'occluded'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    summarize(a.domain, a.arm) if a.summarize else evaluate(a.domain, a.arm)
