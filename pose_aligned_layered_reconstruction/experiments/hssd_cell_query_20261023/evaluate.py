"""Separate SS sampling from expensive mesh decoding; retain all seeds/cases."""
import argparse
import json
import shutil
import numpy as np
import torch
from ...v2.dataset import SceneDataset, select_views
from ...pretrained import load_trellis_model
from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent, SAMPLER_PARAMS
from ...mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from ...evaluate_ss_occupancy import occupancy_metrics
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import ARMS, EVAL_SEEDS, FORMAT, OUT, policy, binding, verify, write
from .model import build, ResidualCFG, observations


def identity(domain, arm):
    contract = policy(domain, 'appearance_control' if arm == 'base' else arm)
    result = {'contract': contract, 'arm': arm, 'seeds': list(EVAL_SEEDS), 'sampler': SAMPLER_PARAMS,
              'ss_steps': 25, 'ss_cfg': 5., 'ss_cfg_interval': [.5, 1.], 'ss_rescale_t': 3.,
              'gt_frame': 'native canonical; no fitted transform', 'weights': 'raw'}
    if arm != 'base':
        result['checkpoint'] = binding(OUT / 'train' / f'{domain}_{arm}' / 'last.pt')
    return result


def root_for(domain, arm):
    return OUT / 'evaluations' / f'{domain}_{arm}'


def verify_identity(domain, arm):
    root, ident = root_for(domain, arm), identity(domain, arm)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != ident:
        raise ValueError('Evaluation identity changed')
    write(path, ident)
    return root, ident


def verify_record(row, source, seed):
    if row['uid'] != source['uid'] or row['seed'] != seed:
        raise ValueError('Unpaired evaluation record')
    verify(row['gt'])
    for key in ('coords', 'mesh'):
        if key in row:
            verify(row[key])


@torch.no_grad()
def sample_ss(domain, arm):
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    root, ident = verify_identity(domain, arm)
    contract = ident['contract']
    dataset = SceneDataset(verify(contract['manifest']), require_labels=arm.startswith('oracle_'), require_ss_target=True)
    if len(dataset) != 16:
        raise ValueError('Expected original16')
    if (root / 'ss_complete.json').exists():
        if json.loads((root / 'ss_complete.json').read_text()) != ident:
            raise ValueError('Changed SS completion identity')
        for index, source in enumerate(dataset.rows):
            for seed in EVAL_SEEDS:
                verify_record(json.loads((root / f'{index:03d}' / f'{seed}_ss.json').read_text()), source, seed)
        return
    sampler, model, saved, bindings = build(contract, device)
    if arm != 'base':
        cp = torch.load(verify(ident['checkpoint']), map_location='cpu', weights_only=False)
        if cp['format'] != FORMAT or cp['contract'] != contract or cp['step'] != contract['steps'] or cp['bindings'] != bindings:
            raise ValueError('Require completed matching residual training')
        model.residual.load_state_dict(cp['residual'], strict=True)
    model.eval().requires_grad_(False)
    decoder, decoder_binding = load_trellis_model(saved['pretrained'], 'ss_dec_conv3d_16l8_fp16')
    decoder = decoder.to(device).eval()
    write(root / 'ss_decoder_binding.json', decoder_binding)
    for index in range(len(dataset)):
        source = dataset[index]
        sample = select_views(source if arm.startswith('oracle_') else observations(source), torch.arange(8))
        folder = root / f'{index:03d}'
        folder.mkdir(exist_ok=True)
        gt = binding(verify(source['target_mesh_binding']))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prepared = model.prepare(sample, device)
        for seed in EVAL_SEEDS:
            path = folder / f'{seed}_ss.json'
            if path.exists():
                verify_record(json.loads(path.read_text()), source, seed)
                continue
            positive = sample['stock_condition'].to(device)
            wrapper = ResidualCFG(model, positive, prepared, enabled=arm != 'base')
            gen = torch.Generator(device=device).manual_seed(seed * 1000003 + index * 1009)
            noise = torch.randn((1, 8, 16, 16, 16), device=device, generator=gen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = sampler.sample(wrapper, noise, cond=positive, neg_cond=torch.zeros_like(positive),
                    steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
                logits = decoder(z.to(next(decoder.parameters()).dtype))
            finite = bool(torch.isfinite(z).all() and torch.isfinite(logits).all())
            coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32) if finite else np.empty((0, 4), np.int32)
            coord_path = folder / f'{seed}_ss.npz'
            np.savez_compressed(coord_path, coords=coords)
            ok = finite and len(coords) > 0
            row = {'uid': source['uid'], 'object_uid': source['object_uid'], 'seed': seed,
                'index': index, 'arm': arm, 'gt': gt, 'coords': binding(coord_path), 'success': bool(ok),
                'positive_calls': wrapper.positive_calls, 'negative_calls': wrapper.negative_calls}
            if ok:
                row['occupancy'] = occupancy_metrics(coords, source['target_coords'].numpy())
            else:
                row['failure'] = 'empty or nonfinite SS support'
            write(path, row)
            print(json.dumps({'domain': domain, **row}), flush=True)
    write(root / 'ss_complete.json', ident)


@torch.no_grad()
def decode_mesh(domain, arm):
    import trimesh
    torch.set_num_threads(2)
    root, ident = verify_identity(domain, arm)
    if json.loads((root / 'ss_complete.json').read_text()) != ident:
        raise ValueError('Require complete matching SS sampling')
    dataset = SceneDataset(verify(ident['contract']['manifest']), require_labels=False)
    install_memory_efficient_mesh()
    device, runtime = torch.device('cuda:0'), None
    for index in range(16):
        sample = select_views(dataset[index], torch.arange(8))
        folder = root / f'{index:03d}'
        source_gt = verify(sample['target_mesh_binding'])
        gt_path = folder / 'gt.ply'
        if gt_path.exists() and binding(gt_path)['sha256'] != binding(source_gt)['sha256']:
            raise ValueError('Changed native GT copy')
        if not gt_path.exists():
            shutil.copyfile(source_gt, gt_path)
        target = trimesh.load(gt_path, force='mesh', process=False)
        for seed in EVAL_SEEDS:
            dest = folder / f'{seed}_candidate.json'
            if dest.exists():
                verify_record(json.loads(dest.read_text()), sample, seed)
                continue
            ss = json.loads((folder / f'{seed}_ss.json').read_text())
            verify_record(ss, sample, seed)
            row = {'uid': sample['uid'], 'object_uid': sample['object_uid'], 'seed': seed,
                'mode': 'candidate', 'gt': binding(gt_path), 'ss_record': binding(folder / f'{seed}_ss.json'),
                'privileged_gt_geometry': arm.startswith('oracle_'), 'support_kind': 'predicted_ss'}
            if not ss['success']:
                row.update(success=False, failure=ss['failure'])
            else:
                if runtime is None:
                    runtime = load_frozen_slat30k(device)
                    write(root / 'slat_binding.json', runtime['binding'])
                with np.load(verify(ss['coords'])) as data:
                    coords = data['coords']
                initial = sparse_noise(coords, master_noise(seed, index, device))
                z, _ = sample_mesh_latent(runtime, sample, initial)
                mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
                if len(mesh.faces) and np.isfinite(mesh.vertices).all():
                    mesh_path = folder / f'{seed}_candidate.ply'
                    mesh.export(mesh_path)
                    mesh = trimesh.load(mesh_path, force='mesh', process=False)
                    row.update(success=True, mesh=binding(mesh_path), surface=surface_metrics(mesh, target), structure=structure_metrics(mesh))
                else:
                    row.update(success=False, failure='empty or nonfinite mesh')
                del z, mesh, initial
            write(dest, row)
            print(json.dumps({'domain': domain, 'arm': arm, **row}), flush=True)
    write(root / 'mesh_complete.json', ident)


def summarize_mesh(domain):
    from ..hssd_pilot_20261020.evaluate import galleries
    roots = {arm: root_for(domain, arm) for arm in ('base', *ARMS)}
    rows = {}
    dataset = SceneDataset(policy(domain, 'appearance_control')['manifest']['path'], require_labels=False)
    for arm, root in roots.items():
        if json.loads((root / 'mesh_complete.json').read_text()) != identity(domain, arm):
            raise ValueError('Incomplete/changed mesh evaluation')
        rows[arm] = []
        for index, source in enumerate(dataset.rows):
            for seed in EVAL_SEEDS:
                row = json.loads((root / f'{index:03d}' / f'{seed}_candidate.json').read_text())
                verify_record(row, source, seed)
                rows[arm].append(row)
    result = {'domain': domain, 'scope': 'training16 diagnostic, no expansion gate or heldout claim',
              'failed': {a: [r for r in rr if not r['success']] for a, rr in rows.items()}, 'comparisons': {}}
    pairs = [(a, 'base') for a in ARMS] + [(a, 'appearance_control') for a in ARMS if a != 'appearance_control']
    pairs += [('oracle_cell', 'oracle_point'), ('predicted_cell', 'predicted_point')]
    for arm, control in pairs:
        records = [{**r, 'mode': mode} for a, mode in ((control, 'parent'), (arm, 'candidate')) for r in rows[a]]
        for a, b in zip(rows[arm], rows[control]):
            if (a['uid'], a['seed'], a['gt']['sha256']) != (b['uid'], b['seed'], b['gt']['sha256']):
                raise ValueError('Unpaired GT across arms')
        result['comparisons'][arm + '_vs_' + control] = (aggregate_mesh_records(records, sorted({r['uid'] for r in records}), EVAL_SEEDS)
            if all(r['success'] for r in records) else {'all_meshes_successful': False})
    for arm in ARMS:
        if not result['failed'][arm] and not result['failed']['base']:
            for index in range(16):
                for seed in EVAL_SEEDS:
                    src = roots['base'] / f'{index:03d}' / f'{seed}_candidate.ply'
                    dst = roots[arm] / f'{index:03d}' / f'{seed}_parent.ply'
                    if dst.exists() and binding(dst)['sha256'] != binding(src)['sha256']:
                        raise ValueError('Changed gallery base')
                    if not dst.exists():
                        shutil.copyfile(src, dst)
            galleries(roots[arm], 16, EVAL_SEEDS)
    write(OUT / f'mesh_summary_{domain}.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['ss', 'mesh', 'summary'])
    p.add_argument('--domain', choices=['clean', 'occluded'], required=True)
    p.add_argument('--arm', choices=['base', *ARMS])
    a = p.parse_args()
    if a.phase == 'summary':
        summarize_mesh(a.domain)
    elif a.arm is None:
        p.error('--arm required')
    elif a.phase == 'ss':
        sample_ss(a.domain, a.arm)
    else:
        decode_mesh(a.domain, a.arm)


if __name__ == '__main__':
    main()
