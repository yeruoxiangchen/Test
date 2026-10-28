"""Matched two-stage training. Completed runs are verified and never retrained."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from ...v2.dataset import SceneDataset, select_views
from ...v2.layer_predictor import GeometryConfig, RayLayerPredictor
from ...v2.train import geometry_forward
from ...v2.losses import multiview_point_loss
from ..hssd_cell_query_20261023.train import atomic_save, fixed_batch, fixed_noise, recover_history
from .common import ARMS, OUT, FORMAT, policy, verify, binding, tensor_hash, read, write, now
from .losses import supervised_loss
from .model import build, observations


def train(kind, arm, steps=None, output=None, lari_path=None, limit_updates=None):
    if limit_updates is not None and limit_updates < 1:
        raise ValueError('limit_updates must be positive')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    contract = policy(kind, arm, steps, lari_path)
    steps = contract['steps']
    output = Path(output) if output else OUT / 'train' / f'{kind}_{arm}'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'run.json').exists() and read(output/'run.json')['contract'] != contract:
            raise ValueError('Changed training policy')
        last, history = output/'last.pt', output/'history.jsonl'
        cp = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
        if cp and (cp['format'] != FORMAT or cp['contract'] != contract or not 0<=cp['step']<=steps):
            raise ValueError('Invalid resume identity')
        dataset = SceneDataset(verify(contract['manifest']), require_labels=kind=='lari', require_ss_target=kind=='ss')
        if len(dataset) != 16:
            raise ValueError('Expected original16')
        data_identity = dataset.identity()
        if cp and cp['data_identity'] != data_identity:
            raise ValueError('Changed data')
        if cp and cp['step'] == steps:
            recover_history(history, steps)
            write(output/'complete.json', {'step':steps, 'contract':contract, 'checkpoint':binding(last),
                  'final_state_sha':tensor_hash(cp['state']), 'data_identity':data_identity})
            return
        torch.manual_seed(contract['seed'])
        if kind == 'lari':
            model = RayLayerPredictor(GeometryConfig(**contract['protocol']['geometry_config'])).to(device)
            trained, bindings = model, {}
        else:
            sampler, model, _, bindings = build(contract, device, checkpointing=True)
            trained = model.residual
        model.train()
        params = list(trained.parameters())
        init_hash = tensor_hash(trained.state_dict())
        frozen_versions = {n:p._version for n,p in model.base.named_parameters()} if kind=='ss' else {}
        optimizer = torch.optim.AdamW(params, lr=contract['lr'], betas=tuple(contract['betas']), weight_decay=contract['weight_decay'])
        start = 0
        if cp:
            if cp['bindings'] != bindings or cp['initial_state_sha'] != init_hash:
                raise ValueError('Initialization or frozen base changed')
            trained.load_state_dict(cp['state'], strict=True)
            optimizer.load_state_dict(cp['optimizer'])
            start = cp['step']
            torch.set_rng_state(cp['rng_cpu'])
            torch.cuda.set_rng_state(cp['rng_cuda'], device)
        run = {'contract':contract, 'data_identity':data_identity, 'bindings':bindings,
               'initial_state_sha':init_hash, 'trainable_parameter_count':sum(p.numel() for p in params)}
        write(output/'run.json', run)
        recover_history(history, start)
        stop = min(steps, start+limit_updates) if limit_updates else steps
        cache, began = {}, time.monotonic()
        for step in range(start, stop):
            index, epoch, offset = fixed_batch(dataset, contract['seed'], step)
            sample = select_views(dataset[index], torch.arange(8))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if kind == 'lari':
                    pred, aux = geometry_forward(model, sample, device)
                    supervised, stats = supervised_loss(pred, sample, arm)
                    cross, counts = multiview_point_loss(pred, sample, aux['cameras'])
                    loss = supervised + .1*cross
                    stats.update(layer_loss=supervised.detach(), cross_view_loss=cross.detach(), **counts)
                else:
                    if index not in cache:
                        cache[index] = {k:v.cpu() for k,v in model.prepare(observations(sample), device).items()}
                    prepared = {k:v.to(device) for k,v in cache[index].items()}
                    target = sample['target'].to(device).float()[None]
                    t, noise = fixed_noise(target, contract['seed'], step)
                    xt, velocity = sampler._get_model_gt(target, t, noise)
                    pred = model(xt, target.new_tensor([t*1000]), sample['stock_condition'].to(device), prepared)
                    loss = F.mse_loss(pred.float(), velocity.float())
                    stats = {'flow_loss':loss.detach(), 'diffusion_time':t}
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('Missing or nonfinite trainable gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = contract['lr']*min(1., (step+1)/contract['warmup'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if kind=='ss' and any(p._version != frozen_versions[n] or p.grad is not None or p.requires_grad
                                  for n,p in model.base.named_parameters()):
                raise RuntimeError('Frozen base changed')
            row = {k:None if v is None else float(v) for k,v in stats.items()}
            row.update(time=now(), step=step+1, kind=kind, arm=arm, uid=sample['uid'], index=index,
                epoch=epoch, offset=offset, loss=float(loss.detach()), gradient_norm=float(norm),
                learning_rate=lr, elapsed_seconds=time.monotonic()-began,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            with history.open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row), flush=True)
            if (step+1)%20==0 or step+1==stop:
                atomic_save(last, {'format':FORMAT, 'step':step+1, **run,
                    'state':{n:p.detach().cpu().clone() for n,p in trained.state_dict().items()},
                    'optimizer':optimizer.state_dict(), 'rng_cpu':torch.get_rng_state(),
                    'rng_cuda':torch.cuda.get_rng_state(device), 'frozen_base_unchanged':True})
        if stop == steps:
            write(output/'complete.json', {'step':steps, 'contract':contract, 'checkpoint':binding(last),
                'final_state_sha':tensor_hash(trained.state_dict()), 'data_identity':data_identity})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['lari','ss']); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--steps', type=int); p.add_argument('--output'); p.add_argument('--lari-path')
    p.add_argument('--limit-updates', type=int)
    a=p.parse_args(); train(a.kind,a.arm,a.steps,a.output,a.lari_path,a.limit_updates)
