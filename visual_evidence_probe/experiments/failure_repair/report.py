import argparse
import math
from collections import defaultdict

import numpy as np

from .config import OUT, PRIOR, PARAMS, folder, registry, read, write, now


def at(d, path):
    for key in path.split('.'):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return float(d) if isinstance(d, (int, float)) and math.isfinite(d) else None


def paired_gain(row, metric, endpoint='accepted', relative=False):
    before = at(row['original_metrics'], metric); after = at(row['metrics'][endpoint], metric)
    if before is None or after is None:
        return None
    gain = after-before if 'fscore' in metric else before-after
    return gain / max(abs(before), 1.e-8) if relative else gain


def stats(pairs):
    grouped = defaultdict(list)
    for uid, value in pairs:
        if value is not None and math.isfinite(value):
            grouped[uid].append(float(value))
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    if not len(means):
        return dict(mean=None, ci95=None, objects=0, cases=0)
    rng = np.random.default_rng(PARAMS['bootstrap_seed'])
    ci = np.quantile(means[rng.integers(len(means), size=(5000, len(means)))].mean(1), [.025, .975]).tolist() if len(means)>1 else None
    return dict(mean=float(means.mean()), ci95=ci, objects=len(means), cases=sum(map(len, grouped.values())),
                improved_objects=int((means>0).sum()), worsened_objects=int((means<0).sum()),
                values_by_object={uid:float(np.mean(v)) for uid,v in sorted(grouped.items())})


def strata(rows, key):
    values = defaultdict(list)
    for r in rows:
        value = at(r, key)
        if value is not None:
            values[r['uid']].append(value)
    ordered = sorted(values, key=lambda uid: (float(np.mean(values[uid])), uid))
    if len(ordered)<2:
        return dict(lower=[], higher=ordered)
    n = len(ordered)//2
    return dict(lower=ordered[:n], higher=ordered[n:])


def summarize(rows):
    quality = strata(rows, 'original_metrics.surface.chamfer_l1')
    suspicion = strata(rows, 'input_score.score')
    populations = dict(all=rows,
        higher_base_error=[r for r in rows if r['uid'] in quality['higher']],
        lower_base_error=[r for r in rows if r['uid'] in quality['lower']],
        higher_input_suspicion=[r for r in rows if r['uid'] in suspicion['higher']],
        lower_input_suspicion=[r for r in rows if r['uid'] in suspicion['lower']])
    metrics = ('surface.chamfer_l1', 'surface.fscore_0p02', 'novel.local_target_depth_mae',
               'novel.local_bad_depth_mae', 'novel.global_union_depth_mae',
               'fixed_error.bad_depth_mae', 'fixed_error.good_depth_mae', 'fixed_error.selected_bad_depth_mae')
    groups = {}
    for name, selected in populations.items():
        d = dict(cases=len(selected), objects=len({r['uid'] for r in selected}),
                 accepted=sum(r['accepted_edit'] for r in selected), endpoints={})
        for ep in ('accepted', 'candidate', 'proposal'):
            d['endpoints'][ep] = {metric:stats([(r['uid'], paired_gain(r, metric, ep)) for r in selected]) for metric in metrics}
            for metric in ('surface.chamfer_l1', 'novel.local_target_depth_mae', 'fixed_error.bad_depth_mae'):
                d['endpoints'][ep][metric + '.relative'] = stats([(r['uid'], paired_gain(r,metric,ep,True)) for r in selected])
        measured = [r for r in selected if paired_gain(r,'fixed_error.bad_depth_mae') is not None]
        d['material_repairs'] = sum(
            (paired_gain(r,'surface.chamfer_l1',relative=True) or 0) >= PARAMS['material_cd_relative_gain'] and
            (paired_gain(r,'fixed_error.bad_depth_mae',relative=True) or 0) >= PARAMS['material_bad_depth_relative_gain'] for r in measured) if measured else None
        d['accepted_cd_regressions'] = sum((paired_gain(r,'surface.chamfer_l1') or 0)<0 for r in selected)
        d['mean_editor_seconds'] = float(np.mean([r['source_elapsed_seconds'] for r in selected])) if selected else None
        groups[name] = d
    return dict(groups=groups, quality_strata=quality, suspicion_strata=suspicion)


def formatted(x, percent=False):
    if x is None:
        return 'NA'
    return f'{x:+.2%}' if percent else f'{x:+.7f}'


def markdown(summary, title):
    lines = [f'# {title}', '', f'时间：{summary["time"]}；案例 {summary["cases"]}/{summary["expected_cases"]}；完整：{summary["complete"]}。', '',
        '主比较：共享输入对齐的原始30K mesh → 完整输入驱动局部SLat纠错。所有回退保留。',
        '高/低错误按对象的初始CD分层，仅用于评估；输入怀疑度在编辑前封存。两个seed先按对象平均。', '',
        '|分组|对象/案例|接受|CD相对改善|局部深度下降|原错误像素误差下降|正确像素误差下降|明显修复案例|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = dict(all='全部',higher_base_error='初始错误较高',lower_base_error='初始错误较低',
                  higher_input_suspicion='输入怀疑度较高',lower_input_suspicion='输入怀疑度较低')
    for name, d in summary['groups'].items():
        e = d['endpoints']['accepted']
        lines.append(f'|{labels[name]}|{d["objects"]}/{d["cases"]}|{d["accepted"]}|'
            + formatted(e['surface.chamfer_l1.relative']['mean'],True)+'|'
            + formatted(e['novel.local_target_depth_mae.relative']['mean'],True)+'|'
            + formatted(e['fixed_error.bad_depth_mae.relative']['mean'],True)+'|'
            + formatted(e['fixed_error.good_depth_mae']['mean'])+f'|{d["material_repairs"]}|')
    lines += ['', '正值为改善；正确像素列为绝对深度误差下降，负值表示损伤。',
              '明显修复：CD至少下降1%且原错误像素误差至少下降5%；这是预先定义的描述阈值，不是显著性结论。',
              '初始错误与收益存在共享基线项，散点相关性仅描述，不单独证明选择性纠错。', '',
              '候选和80步提案、逐对象结果、对象bootstrap区间见 summary.json；原始30K未对齐输出另行保留。',
              '此轮使用扫描物体的渲染输入，不代表真实拍摄或TRELLIS.2对比成绩。']
    if not summary['complete']:
        lines += ['', '**当前为部分进度；分组成员和统计待全队列结束后固定，不作整体结论。**']
    return '\n'.join(lines) + '\n'


def historical():
    outputs = {}
    for split in ('development', 'validation'):
        rows = []
        for r in read(PRIOR / 'registry.json')['rows']:
            if r['split'] != split:
                continue
            for seed in PARAMS['seeds']:
                path = PRIOR / 'objects' / f'{r["index"]:03d}' / str(seed) / 'dense_observed/result.json'
                if path.exists():
                    rows.append(read(path))
        value = dict(time=now(), cases=len(rows), expected_cases=len(rows), complete=True, **summarize(rows))
        value.update(scope='historical post-hoc development analysis; not independent round10 evidence', source=str(PRIOR))
        outputs[split] = value
        (OUT / f'HISTORICAL_{split}.md').write_text(markdown(value, '前七轮既有结果：按初始错误程度复核（事后分析）'))
    write(OUT / 'historical_analysis.json', outputs)


def build():
    rows = []
    for r in registry()['rows']:
        for seed in PARAMS['seeds']:
            path = folder(r['index'],seed) / 'full/result.json'
            if path.exists():
                rows.append(read(path))
    value = dict(time=now(), cases=len(rows), expected_cases=32, complete=len(rows)==32,
                 records=rows, **summarize(rows))
    # Keep the raw-base comparison separate from the local-repair comparison.
    value['end_to_end_cd_gain'] = stats([(r['uid'], at(r['metrics']['raw30k'],'surface.chamfer_l1') -
        at(r['metrics']['accepted'],'surface.chamfer_l1')) for r in rows])
    write(OUT / 'summary.json', value)
    (OUT / 'REPORT.md').write_text(markdown(value, '第十轮：自然生成错误的完整流程修复'))
    print(dict(stage='report', cases=len(rows), complete=value['complete']), flush=True)
    return value


def analysis():
    from .config import STORE
    value=read(OUT/'summary.json')
    if not value['complete']:raise RuntimeError('full cohort required for final analysis')
    groups=value['groups'];all_results=groups['all'];accepted=all_results['endpoints']['accepted']
    def interval(metric,relative=False):
        item=accepted[metric+('.relative' if relative else '')]
        scale=100 if relative else 1
        return f'{item["mean"]*scale:.6g}；对象bootstrap95%区间 {[v*scale for v in item["ci95"]]}'
    lines=['# 第十轮完整结果复核','',f'时间：{now()}。16个新物体×2种子，32/32完成，全部回退计入。','',
        '完整方案相对共享输入对齐的30K基线取得小幅平均改善；当前没有证明对整体错误较大的输出有更强修复能力。','',
        f'- 接受13/32，19个保持基线；接受结果中3个seed案例CD退化。',
        '- CD相对下降（%）：'+interval('surface.chamfer_l1',True),
        '- 新视角局部深度相对下降（%）：'+interval('novel.local_target_depth_mae',True),
        '- 原错误像素误差相对下降（%）：'+interval('fixed_error.bad_depth_mae',True),
        '- 原正确像素误差下降（绝对值，负值为损伤）：'+interval('fixed_error.good_depth_mae'),
        f'- 平均编辑时间{all_results["mean_editor_seconds"]:.2f}秒，不含完整数据准备及生成。','',
        '## 初始错误分层','',
        '初始CD较高的8物体/16案例只接受4例，CD相对变化约-0.02%，配对区间跨零；原错误像素改善约0.07%。',
        '较低错误组接受9/16，CD改善约0.95%，原错误像素改善约1.10%。高输入怀疑度组也没有表现出更高收益。',
        '因此，当前整体平均收益小不能主要解释为大量好样本稀释。高CD分层也不等于“有更大可修复局部错误”：它混合了缺失、整体形状与局部变形，后续需按错误类型检查。','',
        '## 候选与实际交付','',
        '候选的整体CD改善约1.38%、局部深度约5.65%、原错误像素约2.14%；最终交付分别约0.46%、1.93%、0.59%。',
        '接受门确实压缩了收益，但候选的原正确像素也平均受损，不能把候选收益全部当成应放行收益。高错误组候选CD改善约0.35%，其区间仍跨零，说明困难不只在接受门。',
        '0/32满足预定义“CD下降1%且全部原错误像素误差下降5%”阈值；这只是描述标准，全部错误像素的平均也会稀释局部修复，不能解释为没有任何局部成功。','',
        '## 原始base与全流程','',
        f'未对齐原始30K到完整方案的CD绝对改善为{value["end_to_end_cd_gain"]["mean"]}，对象bootstrap95%区间{value["end_to_end_cd_gain"]["ci95"]}，跨零。',
        '局部修复的共享对齐比较与原始base的全流程比较回答不同问题；当前不能宣称全流程相对原始base已经稳定提升。','',
        '## 对后续的约束','',
        '前九/十轮保留为开发记录。TRELLIS.2迁移先建立原生latent、解码梯度与完整输入约束流程；更换生成器本身不能解决证据偏差、局部性或缺失支持问题。',
        '后续真实错误测试应标注可见的局部变形、缺失部件、额外结构、注册失败，分别评价错误区域恢复和正确区域损伤。',
        '本轮为扫描物体的渲染输入；旧方法设计使用过此前开发集，当前16物体未参与本轮方法调参，后续再利用这些结果开发时应将其视为已检查数据。','']
    path=OUT/'ANALYSIS.md';path.write_text('\n'.join(lines))
    audit=STORE/'审计.md';before=audit.read_bytes();marker='## 10.2 第十轮完整结果复核'
    if marker.encode() not in before:
        with audit.open('ab') as stream:stream.write(('\n\n'+marker+'\n\n'+'\n'.join(lines)).encode())
    if not audit.read_bytes().startswith(before):raise RuntimeError('prior audit modified')
    print(dict(analysis=str(path),prior_audit_preserved=True),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--historical',action='store_true');p.add_argument('--analysis',action='store_true');a=p.parse_args()
    if a.analysis:
        analysis()
    elif a.historical:
        historical()
    else:
        build()


if __name__ == '__main__':
    main()
