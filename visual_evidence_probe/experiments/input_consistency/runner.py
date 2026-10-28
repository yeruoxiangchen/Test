import argparse
import gc
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from visual_evidence_probe.common import lock_json
from visual_evidence_probe import runtime
from .config import (BASE, OUT, INDICES, SEEDS, folder, prepare, observations,
                     verify_protocol, read, write, binding, verify, emit, now)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def fit_all(device, context):
    from .refine import fit
    from .policy import THRESHOLDS
    raw = read(OUT / 'raw_decisions.json')['records']
    decisions = []
    for row in raw:
        fits = {key: fit(row['index'], key, context, device) for key in ('a', 'b')}
        selected = row['selected']  # Fixed before fitting: source identity is controlled.
        scores = fits[selected]['chosen']['ious']
        decisions.append(dict(index=row['index'], selected=selected, ious=scores,
            fit=binding(folder(row['index']) / 'refined' / f'{selected}.json'),
            gated={str(t): selected if min(scores) >= t else None for t in THRESHOLDS}))
    lock_json(OUT / 'aligned_decisions.json', dict(uses_gt=False, selection='raw source identity fixed', records=decisions))


@torch.no_grad()
def encode_all(device, context):
    import trimesh
    from ar_ss_flow.shared_object_preprocessing import prepare_shared_object_arrays
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder, _encode_patch_tokens
    from visual_evidence_probe.render import render
    pipeline = build_frozen_dino_encoder(device=device)
    for row in read(OUT / 'aligned_decisions.json')['records']:
        index = row['index']
        path = folder(index) / 'aligned_features.pt'
        if path.exists():
            verify(read(path.with_suffix('.json'))['features'])
            continue
        rec, k, w, _ = observations(index)
        fit = read(verify(row['fit']))
        mesh = trimesh.load(verify(fit['mesh']), force='mesh', process=False)
        images, masks = render(mesh, k[2:], w[2:], context=context)
        prepared = prepare_shared_object_arrays([np.asarray(im) for im in images],
            [m.astype(np.uint8) * 255 for m in masks], resolution=518, foreground_margin=1.1, alpha_threshold=.8)
        cropped_k = torch.from_numpy(prepared.source_to_feature_affines) @ k[2:]
        tokens = torch.cat([_encode_patch_tokens(pipeline, prepared.images[j:j+2]).half().cpu()
                            for j in range(0, 6, 2)])
        root = folder(index) / 'generated_inputs'
        root.mkdir(parents=True, exist_ok=True)
        for j, (raw, cropped) in enumerate(zip(images, prepared.images)):
            raw.save(root / f'{j+2:02d}_raw.png')
            cropped.save(root / f'{j+2:02d}_crop.png')
        np.savez_compressed(root / 'masks.npz', masks=masks)
        rng = np.random.default_rng(20261025)
        points = np.c_[rng.uniform(-.4, .4, (32, 3)), np.ones(32)]
        maximum = 0.
        for old, new, camera, affine in zip(k[2:].numpy(), cropped_k.numpy(), w[2:].numpy(), prepared.source_to_feature_affines):
            xyz = points @ camera.T
            before = xyz[:, :3] @ old.T
            before = before / before[:, 2:]
            after = xyz[:, :3] @ new.T
            after = after / after[:, 2:]
            maximum = max(maximum, float(np.abs(after - before @ affine.T).max()))
        if maximum > 1.e-3:
            raise ValueError(f'crop projection mismatch: {maximum}')
        torch.save(dict(features=tokens, intrinsics=cropped_k), path)
        write(path.with_suffix('.json'), dict(features=binding(path), fit=row['fit'], uses_gt=False,
              dino='FP32; FP16 cache', crop=prepared.geometry_record(), max_projection_error_pixels=maximum))
        emit('encoded_aligned_inputs', index=index, source=row['selected'], projection_error=maximum)
    del pipeline
    cleanup()


def sample_input(index):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    rec, _, w, _ = observations(index)
    original = torch.load(verify(rec['original_features']), map_location='cpu', weights_only=False)
    path = folder(index) / 'aligned_features.pt'
    extra = torch.load(verify(read(path.with_suffix('.json'))['features']), map_location='cpu', weights_only=False)
    features = torch.cat((original['features'][:2], extra['features'])).float()
    return dict(intrinsics=torch.cat((original['intrinsics'][:2], extra['intrinsics'])),
        extrinsics=w, image_size=(518, 518), grid_transform='identity',
        extrinsics_type='w2c', camera_forward_sign=1.,
        **build_masked_dino_contexts(features), visual_patch_features=features)


@torch.no_grad()
def generate_all(device):
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    with FixedNative() as policy:
        ss = runtime.load_ss(device)
        lock_json(OUT / 'ss_binding.json', ss['binding'])
        for index in INDICES:
            sample = sample_input(index)
            for seed in SEEDS:
                path = folder(index) / 'resampled' / f'{seed}_ss.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    start = time.monotonic()
                    runtime.sample_ss(ss, sample, index, seed, path)
                    write(path.with_suffix('.json'), dict(coords=binding(path), elapsed_seconds=time.monotonic()-start))
                emit('aligned_ss_complete', index=index, seed=seed)
        del ss
        cleanup()
        slat = runtime.load_slat(device)
        lock_json(OUT / 'slat_binding.json', slat['binding'])
        for index in INDICES:
            sample = sample_input(index)
            for seed in SEEDS:
                path = folder(index) / 'resampled' / f'{seed}.obj'
                if not path.exists():
                    start = time.monotonic()
                    coords_path = path.with_name(f'{seed}_ss.npz')
                    with np.load(verify(read(coords_path.with_suffix('.json'))['coords'])) as data:
                        coords = data['coords']
                    _, cfg = runtime.mesh_from_support(slat, sample, coords, index, seed, path)
                    write(path.with_suffix('.generation.json'), dict(mesh=binding(path), cfg=cfg,
                          elapsed_seconds=time.monotonic()-start, uses_gt=False))
                else:
                    verify(read(path.with_suffix('.generation.json'))['mesh'])
                emit('aligned_mesh_complete', index=index, seed=seed)
                torch.cuda.empty_cache()
        del slat
        cleanup()
        write(OUT / 'kernel_policy.json', policy.manifest())


def score_all():
    import trimesh
    from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics, structure_metrics
    from pose_aligned_layered_reconstruction.evaluate_ss_occupancy import occupancy_metrics
    targets = read(OUT / 'evaluation_targets.json')['records']
    for row in targets:
        index = row['index']
        gt = trimesh.load(verify(row['gt']), force='mesh', process=False)
        with np.load(verify(row['gt_coords'])) as data:
            target_coords = data['official_coords']
        for key in ('a', 'b', '42', '43'):
            direct = key in ('a', 'b')
            path = folder(index) / ('refined' if direct else 'resampled') / f'{key}.obj'
            result = path.with_suffix('.metrics.json')
            if result.exists():
                verify(read(result)['mesh'])
                continue
            mesh = trimesh.load(path, force='mesh', process=False)
            occupancy = None
            if not direct:
                with np.load(path.with_name(f'{key}_ss.npz')) as data:
                    occupancy = occupancy_metrics(data['coords'], target_coords)
            write(result, dict(index=index, arm='direct_similarity' if direct else 'aligned_source_add',
                  seed={'a': 42, 'b': 43}.get(key, int(key) if key.isdecimal() else None), success=True,
                  surface=surface_metrics(mesh, gt, count=50000, seed=20261025),
                  structure=structure_metrics(mesh), occupancy=occupancy, mesh=binding(path), gt=row['gt'],
                  gt_used_for_inference=False, alignment='two-input-mask proper similarity; no GT ICP'))
        emit('scored', index=index)


def run():
    import nvdiffrast.torch as dr
    verify_protocol()
    start = time.monotonic()
    device = runtime.setup()
    context = dr.RasterizeCudaContext(device=device)
    write(OUT / 'run_started.json', dict(time=now(), gpu_name=torch.cuda.get_device_name(device)))
    fit_all(device, context)
    cleanup()
    encode_all(device, context)
    generate_all(device)
    score_all()
    from .report import report
    report(context=context)
    emit('complete', elapsed_seconds=time.monotonic()-start, new_meshes=8, refined_donors=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'run':
        try:
            run()
        except BaseException as e:
            emit('failed', error=str(e))
            raise
    else:
        from .report import report
        report()


if __name__ == '__main__':
    main()
