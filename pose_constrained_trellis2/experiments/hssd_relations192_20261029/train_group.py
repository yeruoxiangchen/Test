"""Fair group schedules, shared flow time, independent noise, and resumable raw weights."""
import argparse
import os
import random
import time
from pathlib import Path
from . import common as c


def seed_for(*parts): return int(c.digest(list(parts))[:15], 16) % (2**31 - 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=c.CONFIG['arms'], required=True)
    p.add_argument('--stage', choices=['ss', 'shape'], required=True)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-steps', type=int, default=0); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); os.environ.update(c.environment())
    import torch
    from ...conditioning import select_view_indices, build_2d_context, sample_to_device
    from ...models import trainable_named_parameters, trainable_state_dict, load_trainable_state_dict, optimizer_parameter_groups
    from ...runtime import load_pose_conditioned_flow
    from ...train import target_tensor, sample_t, diffuse, flow_mse
    from ...scene_relations.model import attach_relations, bind_neighbors, summarize_state
    from ...scene_relations.data import GroupDataset
    dataset = GroupDataset(a.manifest, a.stage, max_tokens=c.CONFIG['shape_max_tokens'])
    assert len(dataset) and all(g['split'] == 'train' for g in dataset.groups), 'Training must be train-only'
    for g in dataset.groups:
        for row in g['members']:
            for field in ('pose_file', 'ss_target_file' if a.stage == 'ss' else 'shape_target_file'):
                if row.get(field + '_sha256'):
                    assert c.sha(row[field]) == row[field + '_sha256'], f'Changed training input {field}'
    total = a.max_steps or c.CONFIG['epochs'] * len(dataset)
    if total <= 0 or (a.max_steps and not a.smoke): raise ValueError('Step overrides are only for isolated smoke')
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    assert out.is_relative_to(c.OUT)
    contract = {'runtime_identity': c.identity()['identity'], 'manifest_sha256': c.sha(a.manifest),
                'arm': a.arm, 'stage': a.stage, 'smoke': a.smoke,
                'config': c.CONFIG, 'train_groups': len(dataset)}
    c.immutable(out / 'contract.json', contract)
    torch.manual_seed(c.CONFIG['seed']); torch.cuda.manual_seed_all(c.CONFIG['seed'])
    model = load_pose_conditioned_flow(a.stage, device='cuda', model_path=c.models()[a.stage + '_flow'],
                                     lora_rank=8, lora_alpha=16, gradient_checkpointing=True)
    if a.arm == 'relational': model = attach_relations(model)
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, new_lr=c.CONFIG['new_lr'],
        lora_lr=c.CONFIG['lora_lr'], new_weight_decay=.01), betas=(.9, .95), eps=1e-8)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    step, exposures = 0, 0
    checkpoint = out / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        assert saved['contract'] == contract, 'Resume identity changed'
        load_trainable_state_dict(model, saved['model_trainable_state'])
        optimizer.load_state_dict(saved['optimizer']); step = saved['step']; exposures = saved['object_exposures']
        del saved
    if step > total: raise ValueError('Requested budget precedes checkpoint')
    warmup = 1 if a.smoke else c.CONFIG['warmup']
    parameters = list(trainable_named_parameters(model).values())
    started = time.time(); model.train()
    for current in range(step, total):
        order = list(range(len(dataset))); random.Random(seed_for('order', c.CONFIG['seed'], current // len(dataset))).shuffle(order)
        index = order[current % len(dataset)]
        samples = [sample_to_device(s, 'cuda') for s in dataset[index]]
        iteration_seed = seed_for('step', c.CONFIG['seed'], a.stage, current)
        torch.manual_seed(iteration_seed); torch.cuda.manual_seed_all(iteration_seed)
        minimum_views = min(len(s['dino_features']) for s in samples)
        selected = select_view_indices(minimum_views, minimum=2, maximum=4, device='cuda')
        t_value = sample_t(a.stage, torch.device('cuda'))
        unconditional = random.Random(iteration_seed).random() < (0. if a.smoke else c.CONFIG['p_uncond'])
        states, velocities = [], []
        for sample in samples:
            torch.manual_seed(seed_for(iteration_seed, sample['uid'])); torch.cuda.manual_seed_all(seed_for(iteration_seed, sample['uid']))
            state, velocity = diffuse(target_tensor(sample, a.stage, torch.device('cuda')), t_value)
            states.append(state); velocities.append(velocity)
        summaries = [summarize_state(x, s, a.stage, selected) for x, s in zip(states, samples)]
        optimizer.zero_grad(set_to_none=True); losses = []; anchors = []
        for i, (state, sample, velocity) in enumerate(zip(states, samples, velocities)):
            positive = build_2d_context(sample['dino_features'], selected, mode='first')
            context = torch.zeros_like(positive) if unconditional else positive
            used = None if unconditional else sample
            neighbors = [] if unconditional else [s for j, s in enumerate(summaries) if j != i]
            with torch.autocast('cuda', dtype=torch.bfloat16), bind_neighbors(model, state, used, neighbors):
                prediction, stats = model.adapted_prediction(state, torch.tensor([1000 * t_value], device='cuda'),
                    context, used, view_indices=None if unconditional else selected, compare_stock=False)
                loss = flow_mse(prediction, velocity)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite flow loss')
            (loss / len(samples)).backward()
            losses.append(float(loss.detach())); anchors.append(float(stats['zero_init_stock_anchor']))
            del prediction, loss
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('Nonfinite gradients')
        relation_grad = sum(float(p.grad.detach().float().square().sum()) for name, p in model.named_parameters()
                            if '.relation.' in name and p.grad is not None)**.5
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        for group, lr in zip(optimizer.param_groups, base_lrs): group['lr'] = lr * min(1., (current + 1) / warmup)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters): raise RuntimeError('Nonfinite trainable weights')
        exposures += len(samples)
        record = {'step': current + 1, 'total_steps': total, 'epoch': current // len(dataset),
                  'group': dataset.groups[index]['id'], 'loss': sum(losses) / len(losses),
                  't': t_value, 'unconditional': unconditional, 'object_exposures': exposures,
                  'gradient_norm': norm, 'relation_gradient_norm': relation_grad,
                  'initial_stock_anchor': anchors, 'elapsed_seconds': time.time() - started,
                  'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
        if current == step or (current + 1) % 10 == 0 or a.smoke: print(__import__('json').dumps(record), flush=True)
        c.write(out / 'progress.json', record)
        if (current + 1) % c.CONFIG['save_every'] == 0 or current + 1 == total:
            c.atomic_torch_save({'format': 't2_scene_group_v1', 'contract': contract,
                                'model_trainable_state': trainable_state_dict(model),
                                'optimizer': optimizer.state_dict(), 'step': current + 1,
                                'object_exposures': exposures}, checkpoint)
        del samples, states, velocities, summaries
    final = out / 'final.pt'
    if final.exists():
        if not os.path.samefile(checkpoint, final): final.unlink()
    if not final.exists(): os.link(checkpoint, final)
    c.write(out / 'complete.json', {'contract': contract, 'step': total, 'object_exposures': exposures,
                                  'checkpoint': str(final), 'time': c.now()})


if __name__ == '__main__': main()
