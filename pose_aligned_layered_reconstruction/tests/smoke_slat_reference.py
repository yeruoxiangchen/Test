"""Explicit validation-only legacy oracle; never imported by production code."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pose_aligned_layered_reconstruction.frozen_slat import (
    load_frozen_slat30k, FrozenSLatCFG, inference_inputs, master_noise,
    sparse_noise, SAMPLER_PARAMS, project_active_dino,
)
from pose_aligned_layered_reconstruction.paired_dataset import PairedLayerDataset, sha256_file
from pose_aligned_layered_reconstruction.prepare_official_experiment import atomic_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--full_sampling', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    root = Path(__file__).resolve().parents[1] / 'outputs/round1_20260909'
    dataset = PairedLayerDataset(root / 'data_v2/dev64.json', require_labels=False, verify_hashes=False)
    row = dataset.rows[args.index]
    if sha256_file(dataset._resolve(row['sample'])) != row['sha256']:
        raise RuntimeError('sample hash mismatch')
    sample = dataset[args.index]
    runtime = load_frozen_slat30k(device, evaluation_uids=[sample['uid']])
    # The shared base is frozen and identical. The old adapter modules are
    # independently instantiated/loaded, not aliases of the new implementation.
    from pose_aligned_reconstruction.native_slat_genrecon_v2 import NativeSLatGenreconV2Flow
    from pose_aligned_reconstruction.native_slat_genrecon import NativeSLatCalibratedCFGFlow, project_sparse_frustum_dino
    reference = NativeSLatGenreconV2Flow(runtime['model'].flow).to(device).eval()
    reference.aggregator.load_state_dict(runtime['model'].aggregator.state_dict(), strict=True)
    reference.block_condition.load_state_dict(runtime['model'].block_condition.state_dict(), strict=True)
    reference.requires_grad_(False)
    inputs, condition = inference_inputs(sample, device)
    old_sample = dict(inputs, visual_patch_features=inputs['masked_visual_patch_features'])
    coords = np.load(root / f'formal_dev64_eval/coordinates/{args.index:05d}_42_parent.npz')['coords']
    master = master_noise(42, args.index, device)
    current = FrozenSLatCFG(runtime['model'], condition['cond'], inputs)
    old = NativeSLatCalibratedCFGFlow(reference, condition['cond'], old_sample)
    package = Path(__file__).resolve().parents[1]
    source_files = ('frozen_slat.py', 'projection.py', 'view_fusion.py', 'dino_context.py', 'pretrained.py')
    report = dict(index=args.index, uid=sample['uid'], binding=runtime['binding'], cases=[],
        runtime_code_sha256={name: sha256_file(package/name) for name in source_files})
    with torch.autocast('cuda', dtype=torch.bfloat16):
        core = runtime['model'].flow_core
        probe = core.input_layer(sparse_noise(coords, master)).type(core.dtype)
        pt = core.t_embedder(torch.tensor([650.], device=device))
        if core.share_mod:
            pt = core.adaLN_modulation(pt)
        for block in core.input_blocks:
            probe = block(probe, pt.type(core.dtype))
        p1, v1 = project_active_dino(inputs, probe.coords, device)
        p2, v2, _ = project_sparse_frustum_dino(old_sample, probe.coords, device=device)
        c1 = runtime['model'].aggregator(p1, v1)[0]
        c2 = reference.aggregator(p2, v2)[0]
        report['projection'] = dict(feature_max_abs=float((p1-p2).abs().max()), valid_equal=torch.equal(v1,v2),
                                    condition_max_abs=float((c1-c2).abs().max()))
        print(report['projection'], flush=True)
    del probe, p1, p2, c1, c2
    for label, context in [('positive', condition['cond']), ('negative', condition['neg_cond']), ('cached_positive', condition['cond'])]:
        begin = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a = current(sparse_noise(coords, master), torch.tensor([650.], device=device), context)
            b = old(sparse_noise(coords, master), torch.tensor([650.], device=device), context)
        error = float((a.feats - b.feats).abs().max())
        exact = torch.equal(a.coords, b.coords) and torch.equal(a.feats, b.feats)
        report['cases'].append(dict(branch=label, max_abs=error, bit_exact=exact, seconds=time.time()-begin))
        print(report['cases'][-1], flush=True)
        if not exact:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                direct = reference._adapted_core_forward(sparse_noise(coords, master), torch.tensor([650.], device=device),
                    context, current.cache['condition'])
                repeat = current(sparse_noise(coords, master), torch.tensor([650.], device=device), context)
                old_again = old(sparse_noise(coords, master), torch.tensor([650.], device=device), context)
            report['diagnostic'] = dict(new_vs_direct=float((a.feats-direct.feats).abs().max()),
                new_repeat=float((a.feats-repeat.feats).abs().max()), old_repeat=float((b.feats-old_again.feats).abs().max()),
                direct_vs_old=float((b.feats-direct.feats).abs().max()))
            print(report['diagnostic'], flush=True)
            atomic_json(Path(args.output_dir) / 'report.json', dict(report, passed=False))
            raise RuntimeError('frozen SLat port does not reproduce the released flow')
        del a, b
    if args.full_sampling:
        results = []
        for label, wrapper in [('new', FrozenSLatCFG(runtime['model'], condition['cond'], inputs)),
                               ('reference', NativeSLatCalibratedCFGFlow(reference, condition['cond'], old_sample))]:
            begin = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = runtime['sampler'].sample(wrapper, sparse_noise(coords, master),
                    **condition, **SAMPLER_PARAMS, verbose=False).samples
            results.append(value)
            print(dict(sampling=label, seconds=time.time()-begin), flush=True)
        exact = torch.equal(results[0].coords, results[1].coords) and torch.equal(results[0].feats, results[1].feats)
        report['sampling'] = dict(steps=25, bit_exact=exact, max_abs=float((results[0].feats-results[1].feats).abs().max()))
        if not exact:
            atomic_json(Path(args.output_dir) / 'report.json', dict(report, passed=False))
            raise RuntimeError('25-step sampling equivalence failed')
        latent = results[0] * runtime['std'] + runtime['mean']
        mesh = runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
        report['mesh'] = dict(vertices=len(mesh.vertices), faces=len(mesh.faces), bounds=mesh.bounds.tolist())
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise RuntimeError('invalid smoke mesh')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        mesh.export(Path(args.output_dir) / 'port_mesh.ply')
    atomic_json(Path(args.output_dir) / 'report.json', dict(report, passed=True))
    print('SLat30K reference equivalence passed', flush=True)


if __name__ == '__main__':
    main()
