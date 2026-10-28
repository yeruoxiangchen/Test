"""Paired full-trajectory mesh evaluation, with immutable native-frame GT."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent, FrozenSLatCFG, SAMPLER_PARAMS
from ...native_ss import LayeredSSCFGFlow
from ...pretrained import load_trellis_model
from ...mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from ...v2.checkpoints import read_checkpoint, load_state
from ...v2.layer_predictor import GeometryConfig
from ...v2.dataset import SceneDataset, select_views
from ...v2.ss import build_ss
from ...v2.slat import build_slat
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import binding, verify, write_json, event

INPUT_KEYS = ('format', 'uid', 'object_uid', 'raw_visual_patch_features', 'masked_visual_patch_features',
              'visible_target_mask', 'intrinsics', 'extrinsics', 'image_size', 'extrinsics_type',
              'camera_forward_sign', 'layer_camera_y_sign', 'grid_transform')


@torch.no_grad()
def evaluate(a):
    import trimesh
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    dataset = SceneDataset(a.manifest, require_labels=False)
    checkpoint = read_checkpoint(a.checkpoint)
    config = GeometryConfig(**checkpoint['geometry_config'])
    saved = checkpoint['args']
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    run_identity = {'manifest': binding(a.manifest), 'checkpoint': binding(a.checkpoint),
        'mode': a.mode, 'stage': a.stage, 'weights': 'raw', 'seeds': a.seeds, 'sampler': SAMPLER_PARAMS,
        'max_samples': a.max_samples, 'gt_frame': 'native; no fitted transform',
        'evaluation_kind': 'training-object fitting diagnostic, NOT held-out accuracy'}
    identity_file = root / f'{a.mode}_identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != run_identity:
        raise ValueError('Evaluation identity changed')
    write_json(identity_file, run_identity)
    install_memory_efficient_mesh()
    if a.stage == 'slat' and a.mode == 'candidate':
        sampler, model, decoder, bound = build_slat(config, device, a.checkpoint,
            saved['parent_slat30k'], saved['finetune_lora'])
        load_state(model, checkpoint, config, stage='slat', bindings={'parent': bound}, ema=False)
        model.eval()
        runtime = {'sampler': sampler, 'model': model, 'decoder': decoder,
            'mean': torch.tensor(bound['normalization']['mean'], device=device)[None],
            'std': torch.tensor(bound['normalization']['std'], device=device)[None]}
    else:
        runtime = load_frozen_slat30k(device)
    if a.stage == 'ss':
        sampler, ss, bound = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'], False,
            absolute_tolerance=saved.get('ss_absolute_tolerance'), condition_mode=saved['condition_mode'])
        if saved['freeze_geometry']:
            ss.target_lari.requires_grad_(False)
        load_state(ss, checkpoint, config, stage='ss', bindings=bound, ema=False)
        ss.eval()
        ss_decoder, _ = load_trellis_model(saved['pretrained'], 'ss_dec_conv3d_16l8_fp16')
        ss_decoder = ss_decoder.to(device).eval()
    for index in range(min(a.max_samples, len(dataset))):
        source = dataset[index]
        sample = select_views({k: source[k] for k in INPUT_KEYS if k in source}, torch.arange(len(source['intrinsics'])))
        folder = root / f'{index:03d}'
        folder.mkdir(exist_ok=True)
        gt_path = folder / 'gt.ply'
        if not gt_path.exists():
            mesh_binding = source.get('target_mesh_binding')
            if mesh_binding:
                target = trimesh.load(verify(mesh_binding), force='mesh', process=False)
            else:
                # Official control uses official latent-decoded proxy; never
                # substitute this proxy for raw HSSD GT.
                b = source['source_binding']
                verify({'path': b['slat'], 'sha256': b['slat_sha256']})
                with np.load(b['slat']) as d:
                    from trellis.modules import sparse as sp
                    xyz = d['coords'].astype(np.int32)[:, -3:]
                    z = sp.SparseTensor(feats=torch.from_numpy(d['feats'].astype(np.float32)).to(device),
                        coords=torch.from_numpy(np.c_[np.zeros(len(xyz), np.int32), xyz]).to(device))
                runtime['model'].cpu()
                target = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
                del z
            target.export(gt_path)
        target = trimesh.load(gt_path, force='mesh', process=False)
        for seed in a.seeds:
            record_file = folder / f'{seed}_{a.mode}.json'
            if record_file.exists():
                record = json.loads(record_file.read_text())
                verify(record['mesh']); verify(record['gt'])
                continue
            if a.stage == 'ss':
                ss.to(device)
                positive = sample['stock_condition'].to(device)
                wrapper = LayeredSSCFGFlow(ss, positive, sample, use_layer_branch=a.mode == 'candidate')
                generator = torch.Generator(device=device).manual_seed(seed*1000003 + index*1009)
                noise = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = sampler.sample(wrapper, noise, cond=positive, neg_cond=torch.zeros_like(positive),
                        steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
                    logits = ss_decoder(latent.to(next(ss_decoder.parameters()).dtype))
                coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)
                np.savez_compressed(folder / f'{seed}_{a.mode}_ss.npz', coords=coords)
                ss.cpu()
                del latent, logits, wrapper
                torch.cuda.empty_cache()
            else:
                # Fixed native target support isolates SLat learning from SS.
                coords = source['target_coords'].numpy().astype(np.int32)
            if not len(coords):
                raise ValueError('Empty SS support; case is a failure, not excluded')
            runtime['model'].to(device)
            initial = sparse_noise(coords, master_noise(seed, index, device))
            if a.stage == 'slat' and a.mode == 'candidate':
                contexts = {k: [v.to(device) for v in values] for k, values in sample['slat_condition'].items()}
                wrapper = FrozenSLatCFG(runtime['model'], contexts['cond'], sample)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = runtime['sampler'].sample(wrapper, initial, **contexts, **SAMPLER_PARAMS, verbose=False).samples
                z = z * runtime['std'] + runtime['mean']
            else:
                z, _ = sample_mesh_latent(runtime, sample, initial)
            runtime['model'].cpu()
            torch.cuda.empty_cache()
            mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
            mesh_file = folder / f'{seed}_{a.mode}.ply'
            mesh.export(mesh_file)
            record = {'uid': source['uid'], 'object_uid': source['object_uid'], 'seed': seed,
                'mode': a.mode, 'success': True, 'surface': surface_metrics(mesh, target),
                'structure': structure_metrics(mesh), 'mesh': binding(mesh_file), 'gt': binding(gt_path),
                'support_kind': 'native_target' if a.stage == 'slat' else 'predicted_ss',
                'gt_kind': source.get('label_geometry_kind', 'official_latent_proxy')}
            write_json(record_file, record)
            event(stage='mesh_eval', index=index, mode=a.mode, seed=seed, surface=record['surface'])
            del z, mesh, initial
            torch.cuda.empty_cache()
    write_json(root / f'{a.mode}_complete.json', run_identity)


def summarize(path):
    root = Path(path)
    parent = json.loads((root / 'parent_complete.json').read_text())
    candidate = json.loads((root / 'candidate_complete.json').read_text())
    if any(parent[k] != candidate[k] for k in ('manifest', 'checkpoint', 'stage', 'max_samples', 'seeds')):
        raise ValueError('Unpaired evaluation identities')
    ds = SceneDataset(verify(parent['manifest']), require_labels=False)
    uids = [r['uid'] for r in ds.rows[:parent['max_samples']]]
    records = [json.loads(p.read_text()) for p in sorted(root.glob('[0-9][0-9][0-9]/*_parent.json'))]
    records += [json.loads(p.read_text()) for p in sorted(root.glob('[0-9][0-9][0-9]/*_candidate.json'))]
    summary = aggregate_mesh_records(records, uids, seeds=parent['seeds'])
    cd = summary['paired']['chamfer_l1']
    summary['fit_advantage_established'] = cd['mean_improvement'] > 0 and cd['win_objects'] >= len(uids)*.5
    summary['scope'] = 'engineering training-set fit gate; no claim of held-out significance'
    summary['galleries'] = galleries(root, len(uids), parent['seeds'])
    write_json(root / 'summary.json', summary)
    event(stage='eval_summary', output=str(root), fit_advantage=summary['fit_advantage_established'])


@torch.no_grad()
def galleries(root, count, seeds):
    """Every evaluated object, fixed cameras shared by GT/parent/candidate."""
    import trimesh
    from PIL import Image, ImageDraw
    from ...data_v2.render_scenes import pack_scene, rasterize
    from .scenes import look_at
    from .visuals import shade
    device = torch.device('cuda:0')
    side = 384
    focal = side * .5 / np.tan(np.deg2rad(40) * .5)
    k = torch.tensor([[[focal, 0, (side-1)/2], [0, focal, (side-1)/2], [0, 0, 1.]]] * 4,
                     device=device, dtype=torch.float32)
    poses = []
    for angle in np.deg2rad([45, 135, 225, 315]):
        eye = 2 * np.array([np.cos(angle)*np.cos(np.pi/6), np.sin(angle)*np.cos(np.pi/6), .5])
        poses.append(look_at(eye, np.zeros(3), np.array([0., 0., 1.])))
    w = torch.tensor(np.array(poses), device=device, dtype=torch.float32)
    result = []
    for index in range(count):
        folder = root / f'{index:03d}'
        for seed in seeds:
            canvas = Image.new('RGB', (side*3, side*4+36), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), 'GT / parent / candidate | fixed native cameras; no fitted alignment', fill='black')
            paths = [folder / 'gt.ply'] + [folder / f'{seed}_{mode}.ply' for mode in ['parent', 'candidate']]
            for col, path in enumerate(paths):
                mesh = trimesh.load(path, force='mesh', process=False)
                packed = pack_scene([shade(mesh)], [], device)
                rendered = rasterize(packed, k, w, image_size=(side, side), raster_side=side,
                                     layers=1, rgb=True, background_rgb=(1., 1., 1.))
                for view, pixels in enumerate(rendered['rgb']):
                    panel = Image.fromarray((pixels.numpy()*255).round().astype(np.uint8))
                    canvas.paste(panel, (col*side, view*side+36))
                del packed, rendered
            path = folder / f'{seed}_comparison.jpg'
            canvas.save(path, quality=94)
            result.append(binding(path))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest')
    p.add_argument('--checkpoint')
    p.add_argument('--stage', choices=['ss', 'slat'])
    p.add_argument('--mode', choices=['parent', 'candidate'])
    p.add_argument('--output', required=True)
    p.add_argument('--max-samples', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    summarize(args.output) if args.summarize else evaluate(args)
