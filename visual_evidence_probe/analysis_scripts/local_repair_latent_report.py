"""Final latent endpoint report and sparse-support diagnostics, with no GT selection."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from visual_evidence_probe.experiments.local_repair import config
from .local_repair_latent import OUT, MESH_SOURCE


def support_fraction(coordinates, points):
    # Each 64-grid token is subdivided to 4x4x4 cubes, hence 5x5x5 possible corners.
    offset = np.stack(np.meshgrid(*[np.arange(5)] * 3, indexing='ij'), -1).reshape(-1, 3)
    corners = coordinates[:, None, -3:] * 4 + offset
    corner_ids = np.unique(((corners[..., 0] * 257 + corners[..., 1]) * 257 + corners[..., 2]).reshape(-1))
    in_cube = ((points >= -.5) & (points <= .5)).all(1)
    cell = np.floor(np.clip((points + .5) * 256, 0, 256)).astype(np.int64).clip(0, 255)
    offsets = np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing='ij'), -1).reshape(-1, 3)
    query = cell[:, None] + offsets
    ids = (query[..., 0] * 257 + query[..., 1]) * 257 + query[..., 2]
    rows = np.searchsorted(corner_ids, ids)
    present = (rows < len(corner_ids)) & (corner_ids[np.minimum(rows, len(corner_ids) - 1)] == ids)
    present &= in_cube[:, None]
    return dict(all_eight_corners_fraction=float(present.all(1).mean()),
        at_least_one_corner_fraction=float(present.any(1).mean()),
        mean_corner_support_fraction=float(present.mean()), outside_native_cube_fraction=float((~in_cube).mean()))


def summarize():
    expected = [(i, s, m) for i in config.PARAMS['indices'] for s in config.PARAMS['seeds'] for m in config.METHODS]
    records, missing = [], []
    support = []
    for i, seed, method in expected:
        p = OUT / 'objects' / f'{i:03d}' / str(seed) / method / 'result.json'
        if not p.exists():
            missing.append([i, seed, method]); continue
        r = config.read(p)
        if r['status'] == 'no_input_constraints':
            base = config.read(MESH_SOURCE / 'objects' / f'{i:03d}' / str(seed) / 'result.json')['baseline']
            r.update(baseline=base, accepted=base, proposed=base, elapsed_seconds=0.)
            baseline_path = config.BASE / 'objects' / f'{i:03d}' / f'fitted/{seed}.obj'
            destination = p.parent / 'accepted.obj'
            if not destination.exists(): destination.symlink_to(baseline_path)
        else:
            with np.load(MESH_SOURCE / 'objects' / f'{i:03d}' / str(seed) / f'{method}_constraints.npz') as a:
                target, displacement = a['target'], a['displacement']
            with np.load(config.BASE / 'objects' / f'{i:03d}' / f'generated/{seed}_ss.npz') as a:
                coords = a['coords']
            fit = config.read(config.BASE / 'objects' / f'{i:03d}' / f'fitted/{seed}.json')
            inv = np.linalg.inv(fit['chosen']['transform'])
            def native(x): return x @ inv[:3, :3].T + inv[:3, 3]
            sd = dict(index=i, seed=seed, method=method,
                original_surface=support_fraction(coords, native(target - displacement)),
                requested_surface=support_fraction(coords, native(target)),
                initial_sdf_mae=r['initial_target_sdf_mae'], final_sdf_mae=r['curve'][-1]['target_sdf_mae'])
            support.append(sd)
        records.append(r)
    if missing:
        config.write(OUT / 'progress.json', dict(completed=len(records), expected=len(expected), missing=missing))
        raise ValueError(f'latent experiment incomplete: {len(records)}/{len(expected)}')
    rows = {}
    for method in config.METHODS:
        subset = [r for r in records if r['method'] == method]
        gains = {}
        for endpoint in ('accepted', 'proposed'):
            a = np.array([np.mean([r['baseline']['surface']['chamfer_l1'] - r[endpoint]['surface']['chamfer_l1']
                for r in subset if r['index'] == i]) for i in config.PARAMS['indices']])
            f = np.array([np.mean([r[endpoint]['surface']['fscore_0p02'] - r['baseline']['surface']['fscore_0p02']
                for r in subset if r['index'] == i]) for i in config.PARAMS['indices']])
            rng = np.random.default_rng(config.PARAMS['random_seed'])
            boot = a[rng.integers(len(a), size=(5000, len(a)))].mean(1)
            gains[endpoint] = dict(chamfer_l1_improvement=float(a.mean()), fscore_0p02_improvement=float(f.mean()),
                cd_object_deltas=a.tolist(), cd_ci95=np.quantile(boot, [.025, .975]).tolist(),
                wins=int((a > 1.e-8).sum()), losses=int((a < -1.e-8).sum()), ties=int((np.abs(a) <= 1.e-8).sum()))
        rows[method] = dict(accepted_cases=sum(r['edit'] for r in subset),
            optimized_cases=sum(r['status'] == 'complete' for r in subset),
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in subset])), **gains)
    parity = [config.read(OUT / 'latents' / f'{i:03d}_{s}.json') for i in config.PARAMS['indices'] for s in config.PARAMS['seeds']]
    if not all(r['parity'] for r in parity): raise ValueError('latent recreation parity failed')
    summary = dict(time=config.now(), complete=True, records=len(records), methods=rows,
        all16_latent_mesh_parity=True, max_parity_vertex_error=max(r['max_vertex_error'] for r in parity),
        field_support_diagnostics=support,
        all_frozen_rows_exact=all(r.get('frozen_feature_rows_exact', True) for r in records),
        no_new_alignment=True, no_training=True,
        limitations=['Inspected 8-object exploratory cohort', 'Fixed SS support; unreachable target corners default to constant positive SDF.',
            'Corner-SDF proxy omits small vertex deformation and cube weights',
            'Native spconv custom backward not certified bitwise deterministic across processes'])
    config.write(OUT / 'summary.json', summary)
    return summary


def main():
    summary = summarize()
    lines = ['# 第四轮：冻结 SLat 的局部特征优化', '', f'时间：{config.now()}', '',
        '8对象×2seed×4臂，64条状态均已记录。无输入约束时保留原mesh，计入总体。',
        '原始SS支持、30K权重和全局对齐固定。使用同一批局部约束优化选中token；每臂20步。',
        '代理损失是原decoder角点SDF，而实际解码mesh用于最终评价；两者不能混为一谈。', '',
        '|方法|实际优化案例|接受修改|接受后CD改善|完整20步候选CD改善|完整候选F@.02改善|',
        '|---|---:|---:|---:|---:|---:|']
    for m, r in summary['methods'].items():
        lines.append(f'|{m}|{r["optimized_cases"]}/16|{r["accepted_cases"]}/16|{r["accepted"]["chamfer_l1_improvement"]:+.6f}|{r["proposed"]["chamfer_l1_improvement"]:+.6f}|{r["proposed"]["fscore_0p02_improvement"]:+.4f}|')
    lines += ['', '正值为改善；按seed再按对象聚合。oracle两臂使用GT，均仅作诊断。',
        f'16个seed latent均通过原生成mesh的一致性检查，最大顶点差 {summary["max_parity_vertex_error"]:.3g}。',
        '选区外latent特征逐行严格相同；仍单独记录实际解码mesh在选区外的变化。', '',
        '## 实现检查与限制', '',
        '- CUDA grid_sample 3D反向不支持确定性，已换成数值/梯度等价的稀疏八角点插值。',
        '- 原FixedNative策略仅可推理，已限制在重建和端点解码；梯度使用原完整native调优器。',
        '- 原始两次失败及每次修订都保留。未通过修改全局确定性开关绕过错误。',
        '- 固定SS支持外的角点是常数正SDF。目标位置超出支持时，优化现有token无法直接创建那里需要的几何。',
        '- summary.json逐例记录修改前后位置落入现有角点支持的比例；这属于输入/模型支持诊断，不用GT挑选结果。',
        '- 当前角点SDF是纠错代理，实际mesh受deform和cube weights影响。没有建立更可靠的可微mesh约束前，不据代理loss下降宣布修复成功。', '',
        '[完整数值与支持诊断](summary.json)。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    # Preserve original failed execution logs as artifacts on data.
    for unit, label in [('visual-evidence-local-repair-latent-preflight-20261026', 'grid_sample_failure'),
                        ('visual-evidence-local-repair-latent-deterministic-20261026', 'inference_tuner_failure')]:
        result = subprocess.run(['journalctl', '--user', '-u', unit + '.service', '--no-pager'],
            text=True, capture_output=True, check=True)
        p = OUT / 'failure_logs' / f'{label}.txt'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(result.stdout)
    conclusion = config.STORE / '第四次验证结论.md'
    marker = '## SLat局部特征优化补充结果'
    addition = marker + '\n\n' + '\n'.join(lines[2:])
    text = conclusion.read_text()
    if marker not in text:
        with conclusion.open('a') as stream: stream.write('\n\n' + addition)
    audit = config.STORE / '审计.md'
    if marker not in audit.read_text():
        with audit.open('a') as stream: stream.write('\n\n' + addition)
    print(dict(report=str(OUT / 'REPORT.md'), complete=True, records=64), flush=True)


if __name__ == '__main__': main()
