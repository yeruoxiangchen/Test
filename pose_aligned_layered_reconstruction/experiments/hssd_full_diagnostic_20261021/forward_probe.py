"""Measure existing SS conditions and bounded fixed-noise forwards; no updates/sampling."""
import argparse
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from .diagnose import OUT, PILOT, PACKAGE, bind, write, load_prediction, oracle_prediction
from ...projection import build_ss_projection
from ...view_fusion import aggregate_layer_evidence
from ...v2.checkpoints import read_checkpoint, load_state
from ...v2.contracts import LayeredPredictionV2
from ...v2.dataset import SceneDataset, select_views
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import build_ss

VARIANTS = ('parent', 'full', 'view_gate_only', 'injection_only', 'oracle',
            'confidence_one', 'tolerance_005', 'tolerance_010', 'frustum_support')


def rms(x):
    return float(x.float().square().mean().sqrt())


def condition(model, sample, pred, oracle, variant, device):
    if variant == 'parent':
        return model.build_conditions(sample, device=device, use_layer_branch=False)
    if variant == 'no_layer':
        return model.build_conditions(sample, device=device)
    if variant == 'oracle':
        pred = oracle
    if variant == 'confidence_one':
        pred = LayeredPredictionV2(pred.depths, pred.stop_logits, pred.target_logits,
                                  torch.full_like(pred.confidence_logits, 20.))
    if variant == 'injection_only':
        handle = model.view_gate.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
        try:
            return model.build_conditions(sample, device=device, layer_prediction=pred)
        finally:
            handle.remove()
    if variant == 'frustum_support':
        valid = build_ss_projection(sample, device=device)['valid'].float()
        def substitute(encoded, *, appearance_weights, target_support):
            return aggregate_layer_evidence(encoded, appearance_weights=appearance_weights, target_support=valid)
        with patch('pose_aligned_layered_reconstruction.v2.ss.aggregate_layer_evidence', substitute):
            return model.build_conditions(sample, device=device, layer_prediction=pred)
    old = model.match_absolute_tolerance
    try:
        if variant == 'tolerance_005':
            model.match_absolute_tolerance = .05
        elif variant == 'tolerance_010':
            model.match_absolute_tolerance = .1
        result = model.build_conditions(sample, device=device, layer_prediction=pred)
        return (result[0], None, result[2]) if variant == 'view_gate_only' else result
    finally:
        model.match_absolute_tolerance = old


def measure_condition(model, prepared):
    appearance, layer, stats = prepared
    result = {k: float(v) for k, v in stats.items() if torch.is_tensor(v) and v.numel() == 1}
    result['actual_layer_rms'] = 0. if layer is None else rms(layer)
    result['blocks'] = []
    for i, projection in enumerate(model.layer_block_condition.projections):
        parent = model.block_condition(i, appearance)
        injected = torch.zeros_like(parent) if layer is None else projection(layer)
        signal = torch.zeros_like(parent) if layer is None else F.linear(layer, projection.weight, None)
        result['blocks'].append({'index': i, 'parent_appearance_injection_rms': rms(parent),
            'layer_injection_rms': rms(injected), 'layer_weight_times_condition_rms': rms(signal),
            'layer_bias_rms': 0. if layer is None else rms(projection.bias)})
    return result


def forward(model, xt, t, context, prepared):
    ratios = []
    handles = []
    layer = prepared[1]
    for i, block in enumerate(model.flow_core.blocks):
        def hook(module, inputs, block_index=i):
            h = inputs[0]
            injection = torch.zeros_like(h) if layer is None else model.layer_block_condition(block_index, layer).to(h)
            before = rms(h - injection)
            ratios.append({'block': block_index, 'hidden_before_layer_rms': before,
                           'layer_to_hidden_ratio': rms(injection) / max(before, 1.e-12)})
        handles.append(block.register_forward_pre_hook(hook))
    try:
        output = model._core_forward(xt, xt.new_tensor([1000 * t]), context, prepared[0], prepared[1])
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite fixed-noise output')
    return output, ratios


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('SS forward probe requires CUDA; geometry probe separately supports CPU')
    sources = [bind(p) for p in [Path(__file__), Path(__file__).with_name('diagnose.py'),
               PACKAGE / 'v2/ss.py', PACKAGE / 'native_ss.py', PACKAGE / 'view_fusion.py',
               PACKAGE / 'layer_matching.py', PACKAGE / 'v2/checkpoints.py']]
    for domain in args.domains:
        dataset = SceneDataset(PILOT / f'{domain}16.json', require_labels=True, require_ss_target=True)
        for arm in ('full', 'no_layer'):
            checkpoint_path = PILOT / f'{domain}_ss_{arm}/checkpoints/last.pt'
            checkpoint = read_checkpoint(checkpoint_path)
            saved = checkpoint['args']
            config = GeometryConfig(**checkpoint['geometry_config'])
            sampler, model, bindings = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'],
                                               False, absolute_tolerance=saved.get('ss_absolute_tolerance'), condition_mode=arm)
            model.target_lari.requires_grad_(False)
            load_state(model, checkpoint, config, stage='ss', bindings=bindings, ema=False)
            model.eval().requires_grad_(False)
            variants = VARIANTS if arm == 'full' else ('parent', 'no_layer')
            identity = {'sources': sources, 'checkpoint': bind(checkpoint_path), 'manifest': bind(dataset.path),
                'bindings': bindings, 'variants': list(variants), 'forward_indices': list(range(args.forward_count)),
                'times': [.25, .75], 'seed': 20261021, 'dtype': 'bf16 autocast, cached FP32 geometry',
                'scope': 'Post-hoc fixed-noise positive-branch velocity diagnostics, no CFG/mesh performance claim'}
            for index in range(args.start, args.end):
                started = time.time()
                path = OUT / 'conditions' / domain / arm / f'{index:03d}.json'
                geom_path = OUT / 'geometry' / domain / f'{index:03d}.json'
                geo = json.loads(geom_path.read_text())
                if bind(geo['prediction']['path']) != geo['prediction']:
                    raise ValueError('Changed geometry cache')
                record_identity = {**identity, 'geometry_record': bind(geom_path)}
                if path.exists():
                    if json.loads(path.read_text())['identity'] != record_identity:
                        raise ValueError('Diagnostic identity changed')
                    continue
                source = dataset[index]
                sample = select_views(source, torch.arange(len(source['intrinsics'])))
                pred = load_prediction(geo['prediction']['path'], device)
                oracle = oracle_prediction(source, device)
                result = {'identity': record_identity, 'index': index, 'uid': sample['uid'], 'variants': {}}
                parent_predictions = {}
                target = sample['target'].to(device).float()[None]
                generator = torch.Generator(device=device).manual_seed(20261021 + index * 1009)
                noise = torch.randn(target.shape, device=device, generator=generator)
                context = sample['stock_condition'].to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for variant in variants:
                        prepared = condition(model, sample, pred, oracle, variant, device)
                        record = {'condition': measure_condition(model, prepared), 'forwards': []}
                        if index < args.forward_count:
                            for t in [.25, .75]:
                                xt, velocity = sampler._get_model_gt(target, t, noise)
                                output, ratios = forward(model, xt, t, context, prepared)
                                if variant == 'parent':
                                    parent_predictions[t] = output.clone()
                                record['forwards'].append({'time': t,
                                    'flow_loss': float(F.mse_loss(output.float(), velocity.float())),
                                    'output_delta_from_parent_rms': rms(output - parent_predictions[t]),
                                    'block_ratios': ratios})
                        result['variants'][variant] = record
                result['elapsed_seconds'] = time.time() - started
                write(path, result)
                print(json.dumps({'domain': domain, 'arm': arm, 'index': index,
                    'elapsed': result['elapsed_seconds'], 'with_forward': index < args.forward_count}), flush=True)
            del model, sampler, checkpoint
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--domains', nargs='+', choices=['clean', 'occluded'], default=['clean', 'occluded'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=16)
    parser.add_argument('--forward-count', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 16 or not 0 <= args.forward_count <= 4:
        parser.error('bounded original16 cohort, at most first4 objects for forwards')
    run(args)


if __name__ == '__main__':
    main()
