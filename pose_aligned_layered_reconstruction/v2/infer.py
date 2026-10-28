"""V2 layer/SS/mesh inference, with explicit parent comparisons and OBJ output."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..native_ss import LayeredSSCFGFlow
from ..pretrained import load_trellis_model
from ..frozen_slat import (load_frozen_slat30k, sample_mesh_latent, sparse_noise,
                          master_noise, FrozenSLatCFG, SAMPLER_PARAMS)
from .checkpoints import read_checkpoint, load_state, load_geometry, checkpoint_condition_mode
from .dataset import SceneDataset, select_views
from .layer_predictor import GeometryConfig, RayLayerPredictor
from .train import geometry_forward
from .ss import build_ss


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--slat_checkpoint')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--mode', choices=('layers', 'ss', 'mesh'), default='mesh')
    p.add_argument('--parent_ss_only', action='store_true')
    p.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    p.add_argument('--max_samples', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.max_samples, a.steps) < 1:
        raise ValueError('counts must be positive')
    device = torch.device('cuda:0' if a.device == 'cuda' else a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    root = Path(a.output_dir).resolve(); root.mkdir(parents=True, exist_ok=False)
    checkpoint = read_checkpoint(a.checkpoint)
    config = GeometryConfig(**checkpoint['geometry_config'])
    dataset = SceneDataset(a.manifest, require_labels=False)
    if a.mode == 'layers':
        model = RayLayerPredictor(config).to(device).eval()
        load_geometry(model, a.checkpoint, config, a.weights == 'ema')
    else:
        if checkpoint['stage'] != 'ss':
            raise ValueError('SS/mesh inference requires a joint SS checkpoint')
        saved = checkpoint['args']
        sampler, model, binding = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'], False,
                                          absolute_tolerance=saved.get('ss_absolute_tolerance'),
                                          condition_mode=checkpoint_condition_mode(checkpoint))
        if saved['freeze_geometry']:
            model.target_lari.requires_grad_(False)
        if saved['finetune_lora']:
            for name, param in model.flow.named_parameters():
                if 'lora_' in name:
                    param.requires_grad_(True)
        load_state(model, checkpoint, config, stage='ss', bindings=binding, ema=a.weights == 'ema')
        model.eval()
        ss_decoder, _ = load_trellis_model(saved['pretrained'], 'ss_dec_conv3d_16l8_fp16')
        ss_decoder = ss_decoder.to(device).eval().requires_grad_(False)
    runtime = None
    if a.mode == 'mesh':
        if a.slat_checkpoint:
            from .slat import build_slat
            candidate = read_checkpoint(a.slat_checkpoint)
            if candidate['stage'] != 'slat' or candidate['geometry_config'] != checkpoint['geometry_config']:
                raise ValueError('SLat/SS geometry schema mismatch')
            slat_sampler, slat_model, decoder, binding = build_slat(config, device, a.slat_checkpoint,
                candidate['args']['parent_slat30k'], candidate['args']['finetune_lora'])
            load_state(slat_model, candidate, config, stage='slat', bindings={'parent': binding}, ema=a.weights == 'ema')
            slat_model.eval()
            runtime = {'model': slat_model, 'decoder': decoder, 'sampler': slat_sampler, 'binding': binding,
                'mean': torch.tensor(binding['normalization']['mean'], device=device)[None],
                'std': torch.tensor(binding['normalization']['std'], device=device)[None]}
        else:
            runtime = load_frozen_slat30k(device)
    records = []
    for index in range(min(a.max_samples, len(dataset))):
        sample = dataset[index]
        # Inference can read labelled datasets, but labels/latents never enter the model.
        keys = ('format', 'uid', 'object_uid', 'raw_visual_patch_features', 'masked_visual_patch_features',
                'visible_target_mask', 'intrinsics', 'extrinsics', 'image_size', 'extrinsics_type',
                'camera_forward_sign', 'layer_camera_y_sign', 'grid_transform')
        sample = {key: sample[key] for key in keys if key in sample}
        sample = select_views(sample, torch.arange(len(sample['intrinsics'])))
        if sample['grid_transform'] != 'identity':
            raise ValueError('V2 OBJ export currently requires the canonical identity grid')
        folder = root / f'{index:05d}'; folder.mkdir()
        if a.mode == 'layers':
            prediction, _ = geometry_forward(model, sample, device)
            np.savez_compressed(folder / 'layers.npz', depths=prediction.depths.cpu().numpy(),
                target=torch.sigmoid(prediction.target_logits).cpu().numpy(),
                valid=prediction.valid_probabilities().cpu().numpy())
            records.append({'uid': sample['uid'], 'mode': 'layers'})
            continue
        positive = sample['stock_condition'].to(device)
        flow = LayeredSSCFGFlow(model, positive, sample, use_layer_branch=not a.parent_ss_only)
        generator = torch.Generator(device=device).manual_seed(a.seed * 1000003 + index * 1009)
        noise = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            latent = sampler.sample(flow, noise, cond=positive, neg_cond=torch.zeros_like(positive),
                steps=a.steps, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
            logits = ss_decoder(latent.to(next(ss_decoder.parameters()).dtype))
        if not torch.isfinite(latent).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('non-finite SS inference')
        coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)
        np.savez_compressed(folder / 'ss.npz', latent=latent.float().cpu().numpy(), coords=coords)
        record = {'uid': sample['uid'], 'active_voxels': len(coords), 'parent_ss_only': a.parent_ss_only,
                  'condition_mode': checkpoint_condition_mode(checkpoint),
                  'ss_cfg_positive_calls': flow.positive_calls, 'ss_cfg_negative_calls': flow.negative_calls}
        if a.mode == 'mesh':
            if not len(coords):
                raise RuntimeError('empty predicted SS; mesh failure must not be silently filtered')
            initial = sparse_noise(coords, master_noise(a.seed, index, device))
            if a.slat_checkpoint:
                contexts = {k: [v.to(device) for v in values] for k, values in sample['slat_condition'].items()}
                wrapper = FrozenSLatCFG(runtime['model'], contexts['cond'], sample)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = runtime['sampler'].sample(wrapper, initial, **contexts, **SAMPLER_PARAMS, verbose=False).samples
                z = z * runtime['std'] + runtime['mean']
            else:
                z, _ = sample_mesh_latent(runtime, sample, initial)
            if not torch.isfinite(z.feats).all():
                raise FloatingPointError('non-finite SLat')
            # Offload the flow while the stock decoder extracts the surface.
            runtime['model'].to('cpu')
            torch.cuda.empty_cache()
            mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
            if not len(mesh.faces):
                raise RuntimeError('empty output mesh')
            mesh.export(folder / 'mesh_canonical.obj')
            mesh.export(folder / 'mesh_canonical.ply')
            record.update(vertices=len(mesh.vertices), faces=len(mesh.faces))
            runtime['model'].to(device)
        records.append(record)
        (root / 'records.json').write_text(json.dumps(records, indent=2))
    (root / 'complete.json').write_text(json.dumps({'checkpoint': str(Path(a.checkpoint).resolve()),
        'slat_checkpoint': a.slat_checkpoint, 'seed': a.seed, 'records': records}, indent=2))


if __name__ == '__main__':
    main()
