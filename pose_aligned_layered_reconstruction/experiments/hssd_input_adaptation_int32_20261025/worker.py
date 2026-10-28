"""Resume the original sampler/evaluator with an explicit runtime amendment."""
import argparse
import os
import numpy as np
import torch

from .common import *
from .tiled_native import LargeInputNative


def cache_folder(case):
    v, a, o, s = case
    return REPAIR/'latents'/v/a/o['uid']/str(s)


@torch.no_grad()
def cached_sample(case, runtime, sample, initial, inputs):
    from ...frozen_slat import sample_mesh_latent
    from trellis.modules import sparse as sp
    v, a, o, s = case
    folder = cache_folder(case)
    contract = {'runtime_identity': freeze_amendment(),
        'base_identity': freeze_runtime(), 'variant': v, 'arm': a, 'uid': o['uid'], 'seed': s,
        'inputs': inputs, 'ss': binding(base.case_dir(v, a, o['uid'], s)/'ss.json'),
        'slat_binding': runtime['binding']}
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['contract'] != contract:
            raise ValueError('Cached SLat identity differs')
        saved = torch.load(verify(receipt['latent']), map_location='cpu', weights_only=True)
        latent = sp.SparseTensor(coords=saved['coords'].to(initial.device), feats=saved['feats'].to(initial.device))
        calls = receipt['calls']
        event('reuse_exact_slat', variant=v, arm=a, uid=o['uid'], seed=s)
    else:
        event('sample_slat', variant=v, arm=a, uid=o['uid'], seed=s, support=len(initial.coords))
        latent, calls = sample_mesh_latent(runtime, sample, initial)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder/f'sampled_slat.{os.getpid()}.tmp'
        torch.save({'coords': latent.coords.detach().cpu(), 'feats': latent.feats.detach().cpu()}, temporary)
        temporary.replace(folder/'sampled_slat.pt')
        receipt = {'time': now(), 'contract': contract, 'calls': calls,
                   'latent': binding(folder/'sampled_slat.pt')}
        write(receipt_path, receipt)
        event('saved_slat_before_decoder', variant=v, arm=a, uid=o['uid'], seed=s, support=len(latent.coords))
    if (calls != {'positive_calls': 25, 'negative_calls': 19}
            or not torch.equal(latent.coords, initial.coords) or not torch.isfinite(latent.feats).all()):
        raise ValueError('Invalid cached/sampled SLat')
    return latent, calls, receipt


@torch.no_grad()
def mesh():
    from ..hssd_input_adaptation_20261025 import evaluate as e
    runtime_identity = freeze_amendment()
    checks = gates()
    snapshot = read(REPAIR/'original_results.json')
    pending_keys = {tuple(row) for row in snapshot['missing']}
    selected = [j for j in base.new_jobs() if (j[0], j[1], j[2]['uid'], j[3]) in pending_keys]
    current = {}

    class Policy(LargeInputNative):
        def __enter__(self):
            current['policy'] = self
            return super().__enter__()

    def emit(stage, **fields):
        if stage == 'sample_mesh':
            current['case'] = next(j for j in selected if (j[0], j[1], j[2]['uid'], j[3]) ==
                (fields['variant'], fields['arm'], fields['uid'], fields['seed']))
            current['tiled_start'] = len(current['policy'].tiled_calls)
        event(stage, **fields)

    def sample(runtime, sample_value, initial):
        case = current['case']
        _, _, inputs = e.load_input(case[2], case[0], {})
        latent, calls, receipt = cached_sample(case, runtime, sample_value, initial, inputs)
        current['cache_receipt'] = binding(cache_folder(case)/'receipt.json')
        return latent, calls

    def record(path, value):
        if Path(path).name == 'mesh.json':
            value = {**value, 'runtime_identity': runtime_identity,
                'runtime_amendment': binding(REPAIR/'runtime_amendment.json'),
                'regressions': checks, 'sampled_slat_receipt': current['cache_receipt'],
                'native_input_row_tiling': current['policy'].tiled_calls[current['tiled_start']:],
                'all_ss_points_retained': True}
        write(path, value)

    e.FixedNative = Policy
    e.jobs = lambda **kwargs: selected
    e.event = emit
    e.sample_mesh_latent = sample
    e.write = record
    e.mesh()
    freeze_amendment()  # Recheck original source/result preservation after execution.
    amended = []
    for v, a, o, s in selected:
        path = base.case_dir(v, a, o['uid'], s)/'mesh.json'
        row = read(path)
        if row.get('runtime_identity') != runtime_identity:
            raise ValueError('Completed repair case lacks the expected runtime amendment')
        verify(row['mesh']); verify(row['world_mesh']); verify(row['sampled_slat_receipt'])
        amended.append(binding(path))
    write(REPAIR/'mesh_complete.json', {'time': now(), 'runtime_identity': runtime_identity,
        'completed': amended, 'new_meshes': 54, 'original_50_unchanged': True})
    event('repair_mesh_complete', new_meshes=54)


def summary():
    runtime_identity = freeze_amendment()
    checks = gates()
    complete = read(REPAIR/'mesh_complete.json')
    if complete['runtime_identity'] != runtime_identity or len(complete['completed']) != 4:
        raise ValueError('All four repaired slots must complete before summary')
    tiled = []
    for b in complete['completed']:
        row = read(verify(b))
        if row['runtime_identity'] != runtime_identity or row['regressions'] != checks:
            raise ValueError('Unverified runtime change')
        verify(row['runtime_amendment'])
        for call in row['native_input_row_tiling']:
            if call['pairs_expected'] != call['pairs_processed']:
                raise ValueError('Incomplete convolution pairs')
            tiled.append({'uid': row['object_uid'], 'seed': row['seed'], **call})
    from ..hssd_input_adaptation_20261025 import summarize
    summarize.main()
    result = read(OUT/'summary.json')
    result['runtime_amendment'] = binding(REPAIR/'runtime_amendment.json')
    result['runtime_regressions'] = checks
    result['repaired_results'] = complete['completed']
    write(OUT/'summary.json', result)
    with (OUT/'REPORT.md').open('a') as f:
        f.write('\n## 运行修订：大张量 Native 卷积分块\n\n'
            '保留此前50个mesh和全部54个SS；仅补齐4个缺失mesh。超过int32字节寻址范围的192→96卷积按输入行分块，'
            '保留全部SS点、邻域、原center mm和kernel累加顺序，权重、精度、256网格及评估坐标系不变。'
            '小型算子、真实chair latent及接近边界的85267点bench三组回归通过；实际层张量与mesh逐值一致。'
            '这是明确记录的运行实现修订，不代表原始源码未发生任何运行层替换。\n\n'
            '长凳seed44仍保留105283个SS点（GT2652个）；修复运行错误不等于修复模型的过度预测。'
            '修订、回归、全邻接计数和缓存绑定见 [int32_fix_20261025/](int32_fix_20261025/)。\n')
    write(REPAIR/'completed.json', {'time': now(), 'runtime_identity': runtime_identity,
        'summary': binding(OUT/'summary.json'), 'report': binding(OUT/'REPORT.md'),
        'tiled_calls': tiled, 'new_meshes': 54, 'logical_cases': 96, 'review_required': True})
    event('repair_complete_review_required', new_meshes=54, logical_cases=96)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['mesh', 'summary'])
    action = p.parse_args().action
    mesh() if action == 'mesh' else summary()
