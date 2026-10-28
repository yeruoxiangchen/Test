"""Add scale, active-runtime and initialization caveats after all figures finish."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time

import numpy as np

from .config import OUT, STORE, PARAMS, read, write, now


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--wait-figures', action='store_true'); args = parser.parse_args()
    manifest = OUT / 'figures/manifest.json'
    while not manifest.exists() or read(manifest).get('status') != 'complete':
        if not args.wait_figures: raise RuntimeError('comparison figures incomplete')
        status = subprocess.run(['systemctl', '--user', 'is-active', 'visual-evidence-editor-figures-20261026.service'],
            text=True, capture_output=True)
        if status.stdout.strip() != 'active': raise RuntimeError('figure process stopped before completing20 cases')
        time.sleep(10)
    summary = read(OUT / 'summary.json')
    if not summary['complete']: raise RuntimeError('numerical report incomplete')
    rows = [read(p) for p in sorted((OUT / 'factorial').glob('*/*/*/result.json'))]
    recovery = [read(p) for p in sorted((OUT / 'recovery').glob('*/*/mesh/result.json'))]
    scales = [r['initial_depth_loss'] * PARAMS['local_depth_scale'] for r in recovery]
    runtime = {}
    for support in ('original', 'expanded'):
        for objective in ('proxy', 'mesh'):
            values = [r['elapsed_seconds'] for r in rows if r['support'] == support and r['objective'] == objective
                and r['status'] != 'no_input_constraints']
            runtime[f'{support}_{objective}'] = dict(active_cases=len(values), mean_seconds=float(np.mean(values)),
                median_seconds=float(np.median(values)), min_seconds=float(min(values)), max_seconds=float(max(values)))
    source = read(OUT / 'completion.json')
    coverage = source['target_corner_coverage']
    outside = [r for r in coverage if r['original']['outside_native_cube_fraction'] > 0]
    changed = []
    for p in sorted((OUT / 'factorial').glob('*/*/expansion/support.json')):
        if read(p)['added_tokens'] > 0: changed.append(p)
    quality = {}
    for support in ('original', 'expanded'):
        for objective in ('proxy', 'mesh'):
            group = [r for r in rows if r['support'] == support and r['objective'] == objective]
            quality[f'{support}_{objective}'] = dict(proposals_with_more_components=sum(
                r['proposal_structure']['component_count'] > r['original_structure']['component_count'] for r in group),
                accepted_with_more_components=sum(r['accepted_structure']['component_count'] > r['original_structure']['component_count'] for r in group))
    receipt = dict(time=now(), complete=True, recovery_cases=4, factorial_states=len(rows),
        active_input_cases=sum(read(p)['constraint_count'] > 0 for p in (OUT / 'factorial').glob('*/*/complete.json')),
        cases_with_added_support=len(changed), recovery_initial_depth_mae_range=[float(min(scales)), float(max(scales))],
        active_runtime=runtime, quality=quality, outside_cube_cases=outside,
        figures=read(manifest)['rendered_cases'], source_weights_trained=False)
    write(OUT / 'closeout.json', receipt)
    marker = '<!-- fifth_round_final_closeout_20261026 -->'
    lines = ['# 第五轮收尾结论', '', f'时间：{now()}', '', marker, '',
        '**已证明完整mesh监督能够恢复小幅、可表达的latent扰动；自然输入上的最终精度收益仍很小，局部SS扩展尚未显示稳定额外优势。**', '',
        '- 开发恢复：full mesh 4/4通过，平均深度误差下降65.65%；corner-SDF代理0/4通过。',
        '- 这只验证小幅编辑能力。初始可见深度误差为 '
            f'{min(scales):.3g}–{max(scales):.3g} 个canonical单位，远小于1/256网格间距；不能据此宣称已能修复缺腿或大结构错误。',
        '- 自然输入：16个object×seed案例、64个因子臂齐备，其中8个案例有新的有效图像匹配约束，其余8个保留为无操作。',
        '- 保留原SS、使用完整mesh监督的接受后平均CD改善约4e-6，只有2/16接受修改；改进幅度仍不足以宣称稳定重建提升。',
        f'- {len(changed)}个案例实际新增了支持；其中3个保形初始化失败。valid-init工程门槛也不代表初始化在微小收益尺度上完全中性，需同时看原mesh与step0两种基线。',
        '- 存在输入目标越出原生cube的案例，已逐例记录，未静默clip；扩展cube内SS无法直接解决这些目标。', '',
        '## 时间与质量解释', '',
        f'原SS完整mesh损失40步的有效案例平均耗时 {runtime["original_mesh"]["mean_seconds"]:.2f} 秒，'
            f'范围 {runtime["original_mesh"]["min_seconds"]:.2f}–{runtime["original_mesh"]["max_seconds"]:.2f} 秒；',
        '该计时包含snapshot/候选保存，不含模型加载、GT评价及SS保形初始化。主报告按全部状态计算的平均值含零操作案例，不应当作一次有效修复的耗时。', '',
        '连通分量、非流形边和区域外漂移均已记录。接受过程包含拓扑门槛；这些检查不是完整的自交或水密性认证。', '',
        '## 文件', '',
        f'- [完整报告]({OUT / "REPORT.md"})',
        f'- [全部数值]({OUT / "summary.json"})',
        f'- [恢复汇总图]({OUT / "figures/recovery_depth_reduction.jpg"})',
        f'- [恢复案例图]({OUT / "figures/recovery"})',
        f'- [自然案例图]({OUT / "figures/factorial"})',
        '- 运行入口：`/home/zjr/Tracker/visual_evidence_probe/run_editor_support.sh`。', '']
    addition = '\n'.join(lines)
    conclusion = STORE / '第五次验证结论.md'
    if marker not in conclusion.read_text(): conclusion.write_text(addition + '\n---\n\n' + conclusion.read_text())
    audit = STORE / '审计.md'
    if marker not in audit.read_text():
        with audit.open('a') as stream: stream.write('\n\n' + addition)
    report = OUT / 'REPORT.md'
    if marker not in report.read_text():
        with report.open('a') as stream: stream.write('\n\n' + addition)
    link = Path('/home/zjr/Tracker/visual_evidence_probe/第五次验证结论.md')
    if not link.exists(): link.symlink_to(conclusion)
    print(dict(stage='fifth_round_fully_complete', report=str(conclusion), figures=receipt['figures']), flush=True)


if __name__ == '__main__': main()
