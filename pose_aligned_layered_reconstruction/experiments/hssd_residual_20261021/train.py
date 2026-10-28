"""Single-GPU residual-only training with stateless paired noise and strict resume."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ...v2.dataset import SceneDataset, select_views
from .common import ARMS, FORMAT, OUT, STEPS, policy, tensor_hash, verify, write, now
from .model import build


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def fixed_batch(dataset, seed, step):
    epoch, offset = divmod(step, len(dataset))
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + epoch))
    return int(order[offset]), epoch, offset


def fixed_noise(target, seed, step):
    generator = torch.Generator(device=target.device).manual_seed(seed + step * 1009)
    t = float(torch.rand((), generator=generator, device=target.device))
    noise = torch.randn(target.shape, device=target.device, generator=generator, dtype=target.dtype)
    return t, noise


def recover_history(history, start_step):
    if not history.exists():
        if start_step:
            raise ValueError('Checkpoint exists but training history is missing')
        return
    lines = history.read_text().splitlines()
    keep, steps = [], []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError('Corrupt history before final interrupted row')
            continue
        if row['step'] <= start_step:
            keep.append(line)
            steps.append(row['step'])
    if steps != list(range(1, start_step + 1)):
        raise ValueError('History and checkpoint cursor disagree')
    if len(keep) != len(lines):
        history.rename(history.with_name(f'history_interrupted_{time.time_ns()}.jsonl'))
        history.write_text(''.join(line + '\n' for line in keep))


def train(domain, arm, steps=STEPS, output=None):
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    contract = policy(domain, arm, steps)
    output = Path(output) if output else OUT / 'train' / f'{domain}_{arm}'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = output / 'run.json'
        if run_path.exists() and json.loads(run_path.read_text())['contract'] != contract:
            raise ValueError('Training policy/source identity changed; use a new experiment')
        last = output / 'last.pt'
        checkpoint = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
        if checkpoint and (checkpoint['format'] != FORMAT or checkpoint['contract'] != contract):
            raise ValueError('Resume checkpoint contract differs')
        if checkpoint and not 0 <= checkpoint['step'] <= steps:
            raise ValueError('Invalid checkpoint step')
        dataset = SceneDataset(verify(contract['manifest']), require_labels=arm == 'oracle', require_ss_target=True)
        if len(dataset) != 16:
            raise ValueError('Require original16 cohort')
        data_identity = dataset.identity()
        if checkpoint and checkpoint['data_identity'] != data_identity:
            raise ValueError('Resume data identity differs')
        if checkpoint and checkpoint['step'] == steps:
            recover_history(output / 'history.jsonl', steps)
            write(output / 'complete.json', {'time': now(), 'step': steps, 'contract': contract,
                'final_residual_sha': tensor_hash(checkpoint['residual']), 'data_identity': data_identity})
            return
        sampler, model, saved, bindings = build(contract, device, checkpointing=True)
        model.train()
        versions = {n: p._version for n, p in model.base.named_parameters()}
        params = list(model.residual.parameters())
        assert all(not p.requires_grad for p in model.base.parameters())
        optimizer = torch.optim.AdamW(params, lr=contract['lr'], betas=tuple(contract['betas']),
                                      weight_decay=contract['weight_decay'])
        init_hash = tensor_hash(model.residual.state_dict())
        start_step = 0
        if checkpoint:
            if checkpoint['bindings'] != bindings or checkpoint['initial_residual_sha'] != init_hash:
                raise ValueError('Frozen base or residual initialization changed')
            model.residual.load_state_dict(checkpoint['residual'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
        write(run_path, {'contract': contract, 'data_identity': data_identity, 'bindings': bindings,
            'initial_residual_sha': init_hash, 'trainable_parameter_count': sum(p.numel() for p in params),
            'trainable_parameter_names': [n for n, p in model.named_parameters() if p.requires_grad]})
        history = output / 'history.jsonl'
        recover_history(history, start_step)
        cache = {}
        started = time.time()
        for step in range(start_step, steps):
            index, epoch, offset = fixed_batch(dataset, contract['seed'], step)
            sample = select_views(dataset[index], torch.arange(8))
            if len(sample['intrinsics']) != 8:
                raise ValueError('Expected all eight views')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if index not in cache:
                    cache[index] = {k: v.cpu() for k, v in model.prepare(sample, device).items()}
                prepared = {k: v.to(device) for k, v in cache[index].items()}
                target = sample['target'].to(device).float()[None]
                t, noise = fixed_noise(target, contract['seed'], step)
                xt, velocity = sampler._get_model_gt(target, t, noise)
                prediction = model(xt, target.new_tensor([t * 1000]), sample['stock_condition'].to(device), prepared)
                loss = F.mse_loss(prediction.float(), velocity.float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite residual training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = contract['lr'] * min(1., (step + 1) / contract['warmup'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if any(p._version != versions[n] or p.grad is not None for n, p in model.base.named_parameters()):
                raise RuntimeError('Frozen appearance base changed or received gradients')
            row = {'time': now(), 'step': step + 1, 'domain': domain, 'arm': arm, 'uid': sample['uid'],
                'index': index, 'epoch': epoch, 'offset': offset, 'flow_loss': float(loss.detach()),
                'gradient_norm': float(norm), 'diffusion_time': t, 'learning_rate': lr,
                'elapsed_seconds': time.time() - started}
            if step == start_step or (step + 1) % 20 == 0:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    row.update(model.metrics(prepared))
            with history.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
            if (step + 1) % 20 == 0 or step + 1 == steps:
                atomic_save(last, {'format': FORMAT, 'step': step + 1, 'contract': contract,
                    'data_identity': data_identity, 'bindings': bindings, 'initial_residual_sha': init_hash,
                    'residual': {n: v.detach().cpu().clone() for n, v in model.residual.state_dict().items()},
                    'optimizer': optimizer.state_dict(), 'frozen_parameter_versions_unchanged': True})
        write(output / 'complete.json', {'time': now(), 'step': steps, 'contract': contract,
            'final_residual_sha': tensor_hash(model.residual.state_dict()), 'data_identity': data_identity})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['clean', 'occluded'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--steps', type=int, default=STEPS)
    p.add_argument('--output')
    a = p.parse_args()
    train(a.domain, a.arm, a.steps, a.output)
