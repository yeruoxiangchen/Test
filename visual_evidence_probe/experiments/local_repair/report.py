"""Object-level summaries, including rejected proposals and zero-edit cases."""
from .config import OUT, PARAMS, METHODS, read, write, now, STORE
import numpy as np


def paired(records, method, output, section, metric, better):
    values = []
    for index in PARAMS['indices']:
        rows = [r for r in records if r['index'] == index]
        delta = [better * (r['methods'][method][output][section][metric] - r['baseline'][section][metric])
            for r in rows if r['baseline'][section][metric] is not None and r['methods'][method][output][section][metric] is not None]
        if delta:
            values.append(float(np.mean(delta)))
    a = np.asarray(values)
    if not len(a):
        return None
    rng = np.random.default_rng(PARAMS['random_seed'])
    b = a[rng.integers(len(a), size=(5000, len(a)))].mean(1)
    return dict(mean_improvement=float(a.mean()), object_deltas=values, positive_means_better=True,
        wins=int((a > 1.e-10).sum()), losses=int((a < -1.e-10).sum()), ties=int((np.abs(a) <= 1.e-10).sum()),
        object_bootstrap_ci95=np.quantile(b, [.025, .975]).tolist())


def main():
    expected = [(i, s) for i in PARAMS['indices'] for s in PARAMS['seeds']]
    records = []
    missing = []
    for i, seed in expected:
        p = OUT / 'objects' / f'{i:03d}' / str(seed) / 'result.json'
        if p.exists():
            records.append(read(p))
        else:
            missing.append([i, seed])
    if not records:
        raise ValueError('no results')
    summaries = {}
    metrics = [('surface', 'chamfer_l1', -1), ('surface', 'fscore_0p02', 1),
        ('visible', 'fixed_common_depth_mae', -1), ('visible', 'originally_good_depth_mae', -1),
        ('visible', 'heldout_depth_mae', -1)]
    for m in METHODS:
        summaries[m] = dict(accepted_cases=sum(r['methods'][m]['accepted_edit'] for r in records),
            nonzero_proposal_cases=sum(r['methods'][m]['solver']['constraints'] > 0 for r in records),
            total_constraints=sum(r['methods'][m]['solver']['constraints'] for r in records),
            mean_seconds=float(np.mean([r['methods'][m]['elapsed_seconds'] for r in records])),
            accepted={f'{section}/{metric}': paired(records, m, 'accepted', section, metric, sign)
                for section, metric, sign in metrics},
            proposed={f'{section}/{metric}': paired(records, m, 'proposed', section, metric, sign)
                for section, metric, sign in metrics})
    summary = dict(time=now(), complete=not missing, records=len(records), expected=len(expected),
        missing=missing, methods=summaries, cases=[dict(index=r['index'], seed=r['seed'], uid=r['uid'],
            proposal_diagnostic=r['proposal_diagnostic']) for r in records],
        interpretation='Exploratory existing cohort; GT oracle arms privileged; view3 held out from local search only.')
    write(OUT / 'summary.json', summary)
    lines = ['# 第四次验证：输入证据驱动的局部几何修复', '', f'时间：{now()}', '',
        f'完成 {len(records)}/{len(expected)} 个自然重建案例（8对象×2seeds）。', '',
        '复用第三轮已检查的对象，属于开发性验证。输入是扫描物体的渲染图。无模型训练。',
        '第4视图仅对局部深度搜索留出；原生成和全局对齐已经使用它，不能称为未见视图。', '',
        '|方法|接受修改/案例|有约束的案例|接受后CD改善↓|接受后F@.02改善↑|未筛选proposal的CD改善↓|',
        '|---|---:|---:|---:|---:|---:|']
    for m, r in summaries.items():
        cd = r['accepted']['surface/chamfer_l1']['mean_improvement']
        f = r['accepted']['surface/fscore_0p02']['mean_improvement']
        raw = r['proposed']['surface/chamfer_l1']['mean_improvement']
        lines.append(f'|{m}|{r["accepted_cases"]}/{len(records)}|{r["nonzero_proposal_cases"]}|{cd:+.6f}|{f:+.4f}|{raw:+.6f}|')
    lines += ['', '上表改善均按正值更好。CD改善=原始CD−修改后CD。按seed再按对象聚合，保留零修改案例。',
        'oracle_region使用GT选择区域；oracle_direction还使用GT提供方向，均仅用于诊断。', '',
        '## 深度搜索本身是否提供正确方向', '',
        '|对象|seed|有效proposal像素|修改前深度MAE|候选深度MAE|误差改善像素比例|',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        d = r['proposal_diagnostic']
        def fmt(x): return 'NA' if x is None else f'{x:.4f}'
        lines.append(f'|{r["uid"]}|{r["seed"]}|{d["proposals"]}|{fmt(d["before_mae"])}|{fmt(d["after_mae"])}|{fmt(d["improved_fraction"])}|')
    lines += ['', '完整数值、物体级置信区间、正确区域退化与逐候选接受记录见[summary.json](summary.json)。',
        '图像见 figures/；每个案例保存输入proposal、区域、约束、位移、未筛选PLY和接受后OBJ。',
        'SLat可微局部更新的结果与检查另行记录，不把mesh形变当成latent修复。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(dict(report=str(OUT / 'REPORT.md'), complete=not missing, records=len(records)), flush=True)


if __name__ == '__main__':
    main()
