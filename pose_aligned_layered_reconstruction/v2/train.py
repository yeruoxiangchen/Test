"""Runnable LaRI, joint SS and geometry-SLat training with torchrun support."""
import argparse
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..checkpoints import DEFAULT_PARENT_SS30K
from ..frozen_slat import DEFAULT_SLAT
from ..paired_dataset import sha256_file
from .checkpoints import (initialize_ema, update_ema, save_checkpoint, read_checkpoint,
                          load_state, load_geometry, restore_rng)
from .dataset import SceneDataset, select_views, slat_target
from .layer_predictor import GeometryConfig, RayLayerPredictor
from .losses import layer_loss, multiview_point_loss
from .ray_geometry import CameraBundle
from .training_policy import sample_time, learning_rate_factor, validate_resume_policy
from .conditioning import CONDITION_MODES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('lari', 'ss', 'slat'), required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--val_manifest')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--geometry_config', help='JSON containing GeometryConfig fields')
    p.add_argument('--init', help='V2 checkpoint supplying geometry; required for ss/slat')
    p.add_argument('--init_weights', choices=('raw', 'ema'), default='ema')
    p.add_argument('--resume', help='Resume matching stage, optimizer, EMA, cursor and RNG')
    p.add_argument('--parent_ss30k', default=str(DEFAULT_PARENT_SS30K))
    p.add_argument('--parent_slat30k', default=str(DEFAULT_SLAT))
    p.add_argument('--pretrained', default='Stable-X/trellis-vggt-v0-2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=1.e-4)
    p.add_argument('--geometry_lr_scale', type=float, default=.2)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--lr_schedule', choices=('cosine', 'constant'), default='cosine')
    p.add_argument('--lr_schedule_steps', type=int, default=0,
                   help='Cosine horizon; 0 uses max_steps. Allows matched-budget pilot prefixes.')
    p.add_argument('--t_sampling', choices=('uniform', 'logit_normal'), default='uniform')
    p.add_argument('--ss_absolute_tolerance', type=float,
                   help='SS query tolerance override; does not change the geometry predictor')
    p.add_argument('--condition_mode', choices=CONDITION_MODES, default='full',
                   help='SS training/inference condition; single_layer retains only the first scene surface')
    p.add_argument('--cache_frozen_geometry', action='store_true',
                   help='Cache deterministic frozen geometry and supervision metrics during SS')
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--validate_every', type=int, default=500)
    p.add_argument('--val_samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--max_samples', type=int, default=0, help='explicit subset, recorded in checkpoint identity')
    p.add_argument('--amp', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--finetune_lora', action='store_true')
    p.add_argument('--freeze_geometry', action='store_true', help='SS ablation: train adapters only')
    p.add_argument('--cross_view_weight', type=float, default=.1)
    p.add_argument('--layer_weight', type=float, default=1.)
    p.add_argument('--p_uncond', type=float, default=.1)
    p.add_argument('--no_gradient_checkpointing', action='store_true')
    p.add_argument('--trace_phases', action='store_true', help='print model/target/forward/backward phases for diagnostics')
    return p


def geometry_forward(model, sample, device):
    cameras = CameraBundle.from_sample(sample, device)
    prediction, aux = model.predict(raw_patch_features=sample['raw_visual_patch_features'].to(device).float(),
        masked_patch_features=sample['masked_visual_patch_features'].to(device).float(),
        visible_target_mask=sample['visible_target_mask'].to(device).float(),
        intrinsics=cameras.intrinsics, extrinsics=cameras.w2c, extrinsics_type='w2c',
        camera_forward_sign=cameras.forward_sign, image_size=cameras.image_size)
    return prediction, aux


def build_model(args, config, device):
    if args.stage == 'lari':
        return None, RayLayerPredictor(config).to(device), {}, None
    if not args.init and not args.resume:
        raise ValueError('--init is required for ss/slat')
    if args.stage == 'ss':
        from .ss import build_ss
        sampler, model, binding = build_ss(config, device, args.parent_ss30k, args.pretrained,
                                          not args.no_gradient_checkpointing,
                                          absolute_tolerance=args.ss_absolute_tolerance,
                                          condition_mode=args.condition_mode)
        if args.init:
            load_geometry(model.target_lari, args.init, config, ema=args.init_weights == 'ema')
        if args.freeze_geometry:
            model.target_lari.requires_grad_(False)
        if args.finetune_lora:
            for name, p in model.flow.named_parameters():
                if 'lora_' in name:
                    p.requires_grad_(True)
        return sampler, model, binding, None
    from .slat import build_slat
    source = args.init or args.resume
    sampler, model, decoder, binding = build_slat(config, device, source, args.parent_slat30k,
                                                args.finetune_lora, init_ema=args.init_weights == 'ema')
    if not args.no_gradient_checkpointing:
        model.flow_core.use_checkpoint = True
        for block in model.flow_core.blocks:
            block.use_checkpoint = True
    return sampler, model, {'parent': binding}, decoder


def step_loss(args, model, sampler, sample, device, bindings, geometry_cache=None):
    if args.stage == 'slat':
        if args.trace_phases:
            print(json.dumps({'phase': 'slat_target', 'uid': sample['uid']}), flush=True)
        target = slat_target(sample, device, bindings['parent']['normalization'])
        from trellis.modules import sparse as sp
        noise = sp.SparseTensor(feats=torch.randn_like(target.feats), coords=target.coords)
        t = sample_time(args.t_sampling, device)
        xt, velocity = sampler._get_model_gt(target, t, noise)
        unconditional = random.random() < args.p_uncond
        contexts = sample['slat_condition']['neg_cond' if unconditional else 'cond']
        contexts = [value.to(device) for value in contexts]
        if args.trace_phases:
            print(json.dumps({'phase': 'slat_forward', 'coords': len(target.coords)}), flush=True)
        prediction = model(xt, torch.tensor([1000. * t], device=device), contexts,
                           None if unconditional else sample, {})
        loss = F.mse_loss(prediction.feats.float(), velocity.feats.float())
        return loss, {'flow_loss': loss.detach(), 'unconditional': float(unconditional)}
    no_layer = args.stage == 'ss' and getattr(args, 'condition_mode', 'full') == 'no_layer'
    if no_layer:
        # The appearance control never predicts geometry or reads geometry GT.
        # Its optimization objective is only the same SS flow objective.
        prediction = None
        loss = torch.zeros((), device=device)
        metrics = {}
    else:
        geometry = model if args.stage == 'lari' else model.target_lari
        key = sample['uid']
        if geometry_cache is not None and key in geometry_cache:
            prediction, supervised, metrics, consistency, counts = geometry_cache[key]
            metrics = dict(metrics)
        else:
            prediction, aux = geometry_forward(geometry, sample, device)
            supervised, metrics = layer_loss(prediction, sample)
            consistency, counts = multiview_point_loss(prediction, sample, aux['cameras'])
            if geometry_cache is not None:
                if any(p.requires_grad for p in geometry.parameters()):
                    raise ValueError('cannot cache trainable geometry')
                geometry_cache[key] = prediction, supervised, dict(metrics), consistency, counts
        loss = args.layer_weight * supervised + args.cross_view_weight * consistency
        metrics.update({'layer_loss': supervised.detach(), 'cross_view_loss': consistency.detach(), **counts})
    if args.stage == 'ss':
        metrics['target_support_mean'] = 0.
        target = sample['target'].to(device).float()[None]
        t = sample_time(args.t_sampling, device)
        xt, velocity = sampler._get_model_gt(target, t, torch.randn_like(target))
        unconditional = random.random() < args.p_uncond
        context = sample['stock_condition'].to(device)
        if unconditional:
            output, _ = model.adapted_prediction(xt, target.new_tensor([1000. * t]),
                torch.zeros_like(context), None)
        else:
            conditions = model.build_conditions(sample, device=device, layer_prediction=prediction)
            output, _ = model.adapted_prediction(xt, target.new_tensor([1000. * t]), context,
                sample, prepared_conditions=conditions)
            metrics['target_support_mean'] = conditions[2]['target_support_mean'].detach()
        flow = F.mse_loss(output.float(), velocity.float())
        loss = loss + flow
        metrics.update(flow_loss=flow.detach(), unconditional=float(unconditional), diffusion_time=t)
    return loss, metrics


def synchronize_gradients(parameters, world):
    if world == 1:
        return
    # Same parameter order and explicit zero entries for unused CFG branches.
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p in parameters:
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p).clone()
        offset += n


def rank_parameter_hashes(parameters, world):
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().contiguous().cpu()
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    result = [digest.hexdigest()]
    if world > 1:
        result = [None] * world
        dist.all_gather_object(result, digest.hexdigest())
        if len(set(result)) != 1:
            raise RuntimeError('trainable parameters differ across ranks')
    return result


def parameter_inventory(model):
    """Record actual capacity and only the shared SS adapters' initial content."""
    groups = {}
    digests = {prefix: hashlib.sha256() for prefix in
               ('layer_encoder', 'view_gate', 'layer_block_condition')}
    present = set()
    for name, parameter in model.named_parameters():
        prefix = name.split('.')[0]
        counts = groups.setdefault(prefix, {'trainable': 0, 'frozen': 0})
        counts['trainable' if parameter.requires_grad else 'frozen'] += parameter.numel()
        if prefix in digests:
            present.add(prefix)
            value = parameter.detach().contiguous().cpu()
            digests[prefix].update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digests[prefix].update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    counts = {key: sum(group[key] for group in groups.values()) for key in ('trainable', 'frozen')}
    return {**counts, 'by_module': groups}, {key: digests[key].hexdigest() for key in sorted(present)}


def main():
    args = parser().parse_args()
    if min(args.max_steps, args.grad_accum, args.save_every, args.validate_every, args.val_samples) < 1:
        raise ValueError('training/checkpoint intervals must be positive')
    if not 0 <= args.p_uncond < 1 or args.learning_rate <= 0 or args.geometry_lr_scale <= 0:
        raise ValueError('invalid learning rate/unconditional probability')
    if args.resume and args.init:
        raise ValueError('use --resume or --init, not both')
    if args.lr_schedule_steps < 0 or args.warmup_steps < 0:
        raise ValueError('schedule intervals must be nonnegative')
    if args.ss_absolute_tolerance is not None and (args.stage != 'ss' or args.ss_absolute_tolerance <= 0):
        raise ValueError('SS tolerance must be positive and only applies to stage ss')
    if args.condition_mode != 'full' and args.stage != 'ss':
        raise ValueError('condition_mode only applies to SS adapter training')
    if args.condition_mode == 'no_layer' and not args.freeze_geometry:
        raise ValueError('no_layer requires --freeze_geometry; unused geometry cannot be trained')
    if args.cache_frozen_geometry and (args.stage != 'ss' or not args.freeze_geometry):
        raise ValueError('geometry caching requires stage ss and --freeze_geometry')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    if world > 1:
        if device.type != 'cuda':
            raise ValueError('distributed V2 training requires CUDA')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = GeometryConfig(**json.loads(Path(args.geometry_config).read_text())) if args.geometry_config else GeometryConfig()
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    if checkpoint and not args.geometry_config:
        config = GeometryConfig(**checkpoint['geometry_config'])
    if args.init and not args.geometry_config:
        config = GeometryConfig(**read_checkpoint(args.init)['geometry_config'])
    dataset = SceneDataset(args.manifest,
        require_labels=args.stage != 'slat' and args.condition_mode != 'no_layer',
        require_ss_target=args.stage == 'ss')
    if args.max_samples:
        dataset.rows = dataset.rows[:args.max_samples]
        if dataset.base is not None:
            dataset.base.rows = dataset.base.rows[:args.max_samples]
    validation = SceneDataset(args.val_manifest) if args.val_manifest and args.stage != 'slat' else None
    if validation is not None:
        dataset.assert_disjoint(validation)
    identity = dataset.identity()
    identity.update(world_size=world, grad_accum=args.grad_accum, seed=args.seed, max_samples=args.max_samples,
                    validation_sha256=sha256_file(args.val_manifest) if args.val_manifest else None)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        if output.exists() and not args.resume:
            raise FileExistsError(f'use a new output directory: {output}')
        output.mkdir(parents=True, exist_ok=bool(args.resume))
    if world > 1:
        dist.barrier()
    if args.trace_phases:
        print(json.dumps({'phase': 'build_model', 'stage': args.stage}), flush=True)
    sampler, model, bindings, _ = build_model(args, config, device)
    if args.trace_phases:
        print(json.dumps({'phase': 'model_ready'}), flush=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    geometry_params, adapter_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (geometry_params if name.startswith('target_lari.') else adapter_params).append(p)
    groups = []
    for params, scale in ((geometry_params, args.geometry_lr_scale), (adapter_params, 1.)):
        if params:
            groups.append({'params': params, 'lr': args.learning_rate * scale, 'lr_scale': scale})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=.01)
    ema = initialize_ema(model)
    step, cursor = 0, {'epoch': 0, 'offset': 0}
    if checkpoint:
        if checkpoint['data_identity'] != identity:
            raise ValueError('resume dataset/distribution identity differs')
        validate_resume_policy(checkpoint['args'], vars(args))
        load_state(model, checkpoint, config, stage=args.stage, bindings=bindings)
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema = {n: p.to(device) for n, p in checkpoint['ema_state'].items()}
        step, cursor = checkpoint['step'], checkpoint['cursor']
        rng = checkpoint['rng']
        if world > 1:
            rng_file = Path(args.resume).with_name(Path(args.resume).name + f'.rank{rank}.rng.pt')
            rng = torch.load(rng_file, weights_only=False)
        restore_rng(rng)
    else:
        # Identical initialization, independent diffusion noise and CFG dropout.
        random.seed(args.seed + rank * 100003)
        torch.manual_seed(args.seed + rank * 100003)
    if rank == 0:
        capacity, adapter_hashes = parameter_inventory(model)
        prior_config = output / 'run_config.json'
        previous = json.loads(prior_config.read_text()) if args.resume and prior_config.exists() else {}
        initial_adapter_hashes = previous.get('init_adapter_hashes') if args.resume else adapter_hashes
        (output / 'run_config.json').write_text(json.dumps({**vars(args), 'geometry': asdict(config),
            'data_identity': identity, 'bindings': bindings, 'effective_batch': world * args.grad_accum,
            'trainable_parameter_count': capacity['trainable'],
            'frozen_parameter_count': capacity['frozen'], 'parameter_counts': capacity,
            'init_adapter_hashes': initial_adapter_hashes, 'launch_adapter_hashes': adapter_hashes,
            'trainable_parameter_names': [n for n, p in model.named_parameters() if p.requires_grad]}, indent=2))
    start = time.time()
    model.train()
    # Frozen parents stay in eval mode while retaining derivatives wrt conditions.
    if args.stage in ('ss', 'slat'):
        for module in (model.flow, model.aggregator, model.block_condition):
            module.eval()
    if args.stage == 'slat':
        model.geometry.eval()
    if args.stage == 'ss' and args.freeze_geometry:
        model.target_lari.eval()
    geometry_cache = {} if args.cache_frozen_geometry else None
    initial_hashes = rank_parameter_hashes(parameters, world)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = {}
        for _ in range(args.grad_accum):
            order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + cursor['epoch']))
            index = int(order[(cursor['offset'] + rank) % len(dataset)])
            sample = dataset[index]
            sample = select_views(sample, torch.arange(len(sample['intrinsics'])))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and args.amp == 'bf16'):
                loss, metrics = step_loss(args, model, sampler, sample, device, bindings, geometry_cache)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at {sample["uid"]}')
            # A fully unconditional microbatch may have no trainable path when
            # only new conditional adapters are enabled. Other ranks still
            # participate in the same gradient reduction with explicit zeros.
            if loss.requires_grad:
                if args.trace_phases:
                    print(json.dumps({'phase': 'backward'}), flush=True)
                (loss / args.grad_accum).backward()
            metrics['loss'] = loss.detach()
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.) + float(value) / args.grad_accum
            cursor['offset'] += world
            if cursor['offset'] >= len(dataset):
                cursor = {'epoch': cursor['epoch'] + 1, 'offset': 0}
        synchronize_gradients(parameters, world)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        step += 1
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * group['lr_scale'] * learning_rate_factor(args, step)
        optimizer.step()
        update_ema(model, ema)
        if world > 1:
            tensor = torch.tensor(list(accumulated.values()), device=device)
            dist.all_reduce(tensor)
            accumulated = dict(zip(accumulated, (tensor / world).cpu().tolist()))
        row = {'step': step, **cursor, **accumulated, 'gradient_norm': float(norm),
               'object_group_exposures': step * world * args.grad_accum, 'elapsed_seconds': time.time() - start,
               'learning_rates': [g['lr'] for g in optimizer.param_groups]}
        if device.type == 'cuda':
            row['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
        if (validation is not None and step % args.validate_every == 0 and rank == 0
                and args.condition_mode != 'no_layer'):
            geometry = model if args.stage == 'lari' else model.target_lari
            values = []
            with torch.no_grad():
                for i in range(min(args.val_samples, len(validation))):
                    val_sample = validation[i]
                    pred, _ = geometry_forward(geometry, val_sample, device)
                    _, stats = layer_loss(pred, val_sample)
                    values.append(float(stats['depth_mae']))
            row['val_depth_mae'] = sum(values) / len(values)
        if rank == 0:
            print(json.dumps(row), flush=True)
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        if step % args.save_every == 0 or step == args.max_steps:
            for name in (f'step_{step:06d}.pt', 'last.pt'):
                path = output / 'checkpoints' / name
                if rank == 0:
                    save_checkpoint(path, model=model, config=config, stage=args.stage, bindings=bindings,
                        data_identity=identity, optimizer=optimizer, ema=ema, step=step, cursor=cursor, args=args)
                if world > 1:
                    dist.barrier()
                    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                           'cuda': torch.cuda.get_rng_state()}
                    torch.save(rng, str(path) + f'.rank{rank}.rng.pt')
    final_hashes = rank_parameter_hashes(parameters, world)
    if rank == 0:
        (output / 'complete.json').write_text(json.dumps({'step': step, 'stage': args.stage, 'data': identity,
            'initial_rank_parameter_hashes': initial_hashes, 'final_rank_parameter_hashes': final_hashes}, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
