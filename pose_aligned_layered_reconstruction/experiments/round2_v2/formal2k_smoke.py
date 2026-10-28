"""Real-parent, one-GPU acceptance checks for the three formal 2K SS controls.

This writes only to a fresh smoke directory.  It is not formal training, does
not generate meshes, and does not update historical checkpoints or records.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import torch

from pose_aligned_layered_reconstruction.checkpoints import DEFAULT_PARENT_SS30K
from pose_aligned_layered_reconstruction.paired_dataset import sha256_file
from pose_aligned_layered_reconstruction.v2 import train
from pose_aligned_layered_reconstruction.v2.checkpoints import (
    checkpoint_condition_mode, initialize_ema, load_geometry, load_state,
    read_checkpoint, save_checkpoint, update_ema,
)
from pose_aligned_layered_reconstruction.v2.contracts import LayeredPredictionV2
from pose_aligned_layered_reconstruction.v2.dataset import SceneDataset, select_views
from pose_aligned_layered_reconstruction.v2.layer_predictor import GeometryConfig
from pose_aligned_layered_reconstruction.v2.ss import build_ss
from pose_aligned_layered_reconstruction.v2.training_policy import validate_resume_policy

PACKAGE = WORKSPACE / 'pose_aligned_layered_reconstruction'
OLD = PACKAGE / 'outputs/round2_v2_overfit32_20261017'
MODES = ('full', 'single_layer', 'no_layer')
GROUPS = ('layer_encoder.', 'view_gate.', 'layer_block_condition.')
OBSERVATION_KEYS = (
    'format', 'uid', 'object_uid', 'raw_visual_patch_features',
    'masked_visual_patch_features', 'visible_target_mask', 'intrinsics',
    'extrinsics', 'image_size', 'extrinsics_type', 'camera_forward_sign',
    'layer_camera_y_sign', 'grid_transform',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(value):
    print(json.dumps({'utc': now(), **value}, ensure_ascii=False), flush=True)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def parameter_hash(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def forbidden_geometry(*args, **kwargs):
    raise AssertionError('no_layer attempted to execute the geometry predictor')


def inference_sample(sample):
    # Exercise exactly the inference whitelist, with no layer GT or SS target.
    values = {key: sample[key] for key in OBSERVATION_KEYS if key in sample}
    return select_views(values, torch.arange(len(values['intrinsics'])))


def max_condition_delta(first, second):
    return max(float((a.float() - b.float()).abs().max())
               for a, b in zip(first[:2], second[:2]))


def forward_reference(model, sample, device, seed):
    positive = sample['stock_condition'].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        conditions = model.build_conditions(sample, device=device)
        prediction, _ = model.adapted_prediction(noise, noise.new_tensor([500.]),
            positive, sample, prepared_conditions=conditions)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('non-finite checkpoint inference prediction')
    return tuple(value.detach().clone() for value in conditions[:2]), prediction.detach().clone()


def verify_resume_rejection(checkpoint, model, config, bindings):
    saved = checkpoint['args']
    validate_resume_policy(saved, dict(saved))
    for other in MODES:
        if other == saved['condition_mode']:
            continue
        altered = {**saved, 'condition_mode': other}
        with patch.object(model, 'condition_mode', other):
            try:
                # The real resume path performs both checks before restoring
                # optimizer state.  Mode validation belongs to load_state.
                validate_resume_policy(saved, altered)
                load_state(model, checkpoint, config, stage='ss', bindings=bindings)
            except ValueError as error:
                if 'condition_mode' not in str(error):
                    raise AssertionError('resume rejection did not identify condition_mode') from error
            else:
                raise AssertionError('resume accepted a changed conditioning mode')


def run_mode(mode, samples, dataset_identity, options, config, output, device):
    started = time.monotonic()
    seed_all(options.seed)
    emit({'phase': 'build_real_parent', 'condition_mode': mode})
    sampler, model, bindings = build_ss(config, device, options.parent_ss,
        options.pretrained, True, absolute_tolerance=options.absolute_tolerance,
        condition_mode=mode)
    load_geometry(model.target_lari, options.geometry_init, config, ema=False)
    model.target_lari.requires_grad_(False)
    model.train()
    for module in (model.flow, model.aggregator, model.block_condition, model.target_lari):
        module.eval()
    named = dict(model.named_parameters())
    trainable = {name: value for name, value in named.items() if value.requires_grad}
    allowed = GROUPS + (('appearance_control.',) if mode == 'no_layer' else ())
    if not trainable or any(not name.startswith(allowed) for name in trainable):
        raise AssertionError('unexpected parameter made trainable')
    frozen_before = parameter_hash((name, value) for name, value in named.items()
                                   if not value.requires_grad)
    initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
    shared_hash = parameter_hash((name, value) for name, value in trainable.items()
                                if name.startswith(GROUPS))
    args = argparse.Namespace(stage='ss', condition_mode=mode, trace_phases=False,
        t_sampling='uniform', p_uncond=0., layer_weight=1., cross_view_weight=.1,
        freeze_geometry=True, finetune_lora=False, init_weights='raw',
        learning_rate=1.e-4, geometry_lr_scale=.2, warmup_steps=0,
        max_steps=options.steps, lr_schedule='constant', lr_schedule_steps=0,
        ss_absolute_tolerance=options.absolute_tolerance, amp='bf16',
        cache_frozen_geometry=False, parent_ss30k=options.parent_ss,
        pretrained=options.pretrained)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.learning_rate,
                                  betas=(.9, .95), weight_decay=.01)
    ema = initialize_ema(model)
    maxima = {prefix: 0. for prefix in allowed}
    rows = []
    noise_hashes = []
    get_model_gt = sampler._get_model_gt
    def capture_model_gt(target, diffusion_time, noise):
        noise_hashes.append(parameter_hash([('noise', noise)]))
        return get_model_gt(target, diffusion_time, noise)
    torch.cuda.reset_peak_memory_stats(device)
    with ExitStack() as stack:
        if mode == 'no_layer':
            stack.enter_context(patch.object(model.target_lari, 'forward', forbidden_geometry))
            stack.enter_context(patch.object(model.target_lari, 'predict', forbidden_geometry))
            stack.enter_context(patch.object(model, 'predict_layers', forbidden_geometry))
        stack.enter_context(patch.object(train, 'sample_time', lambda policy, where: .5))
        stack.enter_context(patch.object(sampler, '_get_model_gt', capture_model_gt))
        for step in range(options.steps):
            # Same clean/occluded sequence and exact diffusion noise across arms.
            seed_all(options.seed + 1000 + step)
            sample = samples[step % len(samples)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = train.step_loss(args, model, sampler, sample, device, bindings)
            if not torch.isfinite(loss) or not loss.requires_grad:
                raise AssertionError('flow objective is non-finite or has no adapter gradient path')
            loss.backward()
            group_maxima = {}
            for prefix in allowed:
                gradients = [p.grad for n, p in trainable.items()
                             if n.startswith(prefix) and p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('non-finite gradient in ' + prefix)
                observed = max((float(g.detach().abs().max()) for g in gradients), default=0.)
                maxima[prefix] = max(maxima[prefix], observed)
                group_maxima[prefix] = observed
            if any(p.grad is not None for p in named.values() if not p.requires_grad):
                raise AssertionError('frozen parent or geometry received gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            update_ema(model, ema)
            row = {'step': step + 1, 'uid': sample['uid'], 'condition_mode': mode,
                   'loss': float(loss.detach()), 'flow_loss': float(metrics['flow_loss']),
                   'diffusion_time': float(metrics['diffusion_time']),
                   'diffusion_noise_seed': options.seed + 1000 + step,
                   'diffusion_noise_sha256': noise_hashes[-1],
                   'gradient_norm': float(norm), 'gradient_max_by_group': group_maxima}
            rows.append(row)
            emit({'phase': 'adapter_update', **row})
            del loss, metrics
        if any(value <= 0 for value in maxima.values()):
            raise AssertionError('a required adapter group never received a nonzero gradient: ' + repr(maxima))
        updates = {prefix: max(float((p.detach().cpu() - initial[n]).abs().max())
                              for n, p in trainable.items() if n.startswith(prefix))
                   for prefix in allowed}
        if any(value <= 0 for value in updates.values()):
            raise AssertionError('a required adapter group did not update')
        frozen_after = parameter_hash((name, value) for name, value in named.items()
                                      if not value.requires_grad)
        if frozen_before != frozen_after:
            raise AssertionError('frozen parameters changed during adapter smoke')
        model.eval()
        observed = inference_sample(samples[1])
        expected_conditions, expected_prediction = forward_reference(model, observed, device, options.seed + 77)
        behavior = {'no_layer_geometry_forbidden_hook_installed': mode == 'no_layer'}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if mode == 'no_layer':
                modified = {**observed, 'raw_visual_patch_features':
                            observed['raw_visual_patch_features'] + .5}
                changed = model.build_conditions(modified, device=device)
                delta = max_condition_delta(expected_conditions, changed)
                if delta <= 0:
                    raise AssertionError('no_layer condition ignores observed raw image features')
                behavior['raw_feature_condition_delta'] = delta
            if mode == 'single_layer':
                prediction, _ = train.geometry_forward(model.target_lari, observed, device)
                altered = LayeredPredictionV2(prediction.depths.clone(), prediction.stop_logits.clone(),
                    prediction.target_logits.clone(), prediction.confidence_logits.clone())
                altered.depths[:, 1:] += 10.
                altered.target_logits[:, 1:] *= -1.
                altered.confidence_logits[:, 1:] *= -1.
                original = model.build_conditions(observed, device=device, layer_prediction=prediction)
                changed = model.build_conditions(observed, device=device, layer_prediction=altered)
                delta = max_condition_delta(original, changed)
                if delta != 0:
                    raise AssertionError('single_layer condition depends on later intersection fields')
                behavior['later_intersection_condition_delta'] = delta
        checkpoint_path = output / mode / 'checkpoint.pt'
        save_checkpoint(checkpoint_path, model=model, config=config, stage='ss', bindings=bindings,
            data_identity=dataset_identity, optimizer=optimizer, ema=ema, step=options.steps,
            cursor={'epoch': options.steps // 2, 'offset': options.steps % 2}, args=args)
        checkpoint = read_checkpoint(checkpoint_path)
        if checkpoint_condition_mode(checkpoint) != mode:
            raise AssertionError('checkpoint lost conditioning mode metadata')
        verify_resume_rejection(checkpoint, model, config, bindings)
        for other in MODES:
            if other == mode:
                continue
            with patch.object(model, 'condition_mode', other):
                try:
                    load_state(model, checkpoint, config, stage='ss', bindings=bindings)
                except ValueError as error:
                    if 'condition_mode' not in str(error):
                        raise AssertionError('wrong-mode checkpoint failed for unrelated reason') from error
                else:
                    raise AssertionError('checkpoint loaded into a different conditioning mode')
        # Corrupt every saved trainable tensor, then require the loader to restore
        # the exact serialized values and the inference-whitelist conditions.
        with torch.no_grad():
            for p in trainable.values():
                p.add_(.01)
        load_state(model, checkpoint, config, stage='ss', bindings=bindings, ema=False)
        for name, saved in checkpoint['model_state'].items():
            if not torch.equal(named[name].detach().cpu(), saved):
                raise AssertionError('checkpoint tensor failed exact restore: ' + name)
        restored_conditions, restored_prediction = forward_reference(model, observed, device, options.seed + 77)
        condition_delta = max_condition_delta(expected_conditions, restored_conditions)
        prediction_delta = float((expected_prediction.float() - restored_prediction.float()).abs().max())
        if condition_delta > 1.e-6 or prediction_delta > 1.e-5:
            raise AssertionError(f'checkpoint inference roundtrip differs: {condition_delta}, {prediction_delta}')
    result = {'condition_mode': mode, 'passed': True, 'steps': options.steps,
        'trainable_parameter_count': sum(p.numel() for p in trainable.values()),
        'trainable_parameters_by_group': {prefix: sum(p.numel() for n, p in trainable.items()
            if n.startswith(prefix)) for prefix in allowed},
        'trainable_parameter_names': sorted(trainable),
        'shared_adapter_initial_sha256': shared_hash,
        'frozen_parameter_sha256_before': frozen_before,
        'frozen_parameter_sha256_after': frozen_after,
        'gradient_max_by_group': maxima, 'parameter_update_max_by_group': updates,
        'parent_binding': bindings['parent'], 'history': rows,
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256_file(checkpoint_path),
        'condition_roundtrip_max_abs': condition_delta,
        'prediction_roundtrip_max_abs': prediction_delta,
        'changed_mode_resume_rejected': True, 'changed_mode_checkpoint_load_rejected': True,
        'behavior': behavior, 'elapsed_seconds': time.monotonic() - started,
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20}
    write_json(output / mode / 'result.json', result)
    emit({'phase': 'arm_passed', 'condition_mode': mode,
          'trainable_parameter_count': result['trainable_parameter_count']})
    del model, sampler, optimizer, ema, named, trainable, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manifest', default=str(OLD / 'train96.json'))
    parser.add_argument('--geometry-init', default=str(OLD / 'lari/checkpoints/step_002000.pt'))
    parser.add_argument('--parent-ss', default=str(DEFAULT_PARENT_SS30K))
    parser.add_argument('--pretrained', default='Stable-X/trellis-vggt-v0-2')
    parser.add_argument('--absolute-tolerance', type=float, default=.02)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261018)
    options = parser.parse_args()
    if not 2 <= options.steps <= 4:
        parser.error('--steps must be 2..4: smoke is not formal training')
    output = Path(options.output_dir).resolve()
    if not output.is_relative_to(PACKAGE / 'outputs'):
        parser.error('--output-dir must be under this package outputs directory')
    output.mkdir(parents=True, exist_ok=False)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('select exactly one accessible GPU with CUDA_VISIBLE_DEVICES')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        dataset = SceneDataset(options.manifest, require_labels=True, require_ss_target=True)
        selected = [0, 32]
        samples = [select_views(dataset[index], torch.arange(len(dataset[index]['intrinsics'])))
                   for index in selected]
        if (samples[0]['object_uid'] != samples[1]['object_uid']
                or ':occlusion:' in samples[0]['uid'] or ':occlusion:' not in samples[1]['uid']):
            raise AssertionError('fixed smoke pair is not one clean/occluded object')
        source = read_checkpoint(options.geometry_init)
        if source['stage'] != 'lari' or source['step'] != 2000:
            raise AssertionError('smoke must use the existing LaRI step2000, never a formal checkpoint')
        config = GeometryConfig(**source['geometry_config'])
        del source
        identity = {'source_manifest': str(Path(options.manifest).resolve()),
                    'source_manifest_sha256': sha256_file(options.manifest),
                    'selected_indices': selected, 'uids': [sample['uid'] for sample in samples],
                    'object_uids': sorted({sample['object_uid'] for sample in samples})}
        code_files = [Path(__file__), PACKAGE / 'v2/train.py', PACKAGE / 'v2/ss.py',
                      PACKAGE / 'v2/conditioning.py', PACKAGE / 'v2/checkpoints.py',
                      PACKAGE / 'v2/training_policy.py']
        protocol = {'started_utc': now(), 'formal_training': False, 'steps_per_arm': options.steps,
            'condition_modes': list(MODES), 'data': identity,
            'geometry_init': str(Path(options.geometry_init).resolve()),
            'geometry_init_sha256': sha256_file(options.geometry_init), 'geometry_weights': 'raw',
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
            'fixed_diffusion_time': .5, 'same_microbatch_order_and_noise': True,
            'seed': options.seed, 'code_sha256': {str(path): sha256_file(path) for path in code_files}}
        write_json(output / 'protocol.json', protocol)
        results = [run_mode(mode, samples, identity, options, config, output, device) for mode in MODES]
        if len({r['shared_adapter_initial_sha256'] for r in results}) != 1:
            raise AssertionError('shared adapter initialization differs across controls')
        if len({r['parent_binding']['sha256'] for r in results}) != 1:
            raise AssertionError('parent SS initialization differs across controls')
        for step in range(options.steps):
            if len({r['history'][step]['diffusion_noise_sha256'] for r in results}) != 1:
                raise AssertionError('diffusion noise differs across controls')
        for path, expected in protocol['code_sha256'].items():
            if sha256_file(path) != expected:
                raise AssertionError('smoke source changed while executing: ' + path)
        summary = {'passed': True, 'completed_utc': now(), 'formal_training': False,
            'protocol_sha256': sha256_file(output / 'protocol.json'), 'results': results,
            'shared_adapter_initialization_equal': True,
            'limits': ['three optimizer updates are interface checks, not reconstruction evidence',
                       'frozen SLat and mesh decoder are not executed by this SS acceptance check']}
        write_json(output / 'complete.json', summary)
        emit({'phase': 'complete', 'passed': True, 'output': str(output)})
    except BaseException as error:
        write_json(output / 'failure.json', {'utc': now(), 'error': repr(error),
                                           'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
