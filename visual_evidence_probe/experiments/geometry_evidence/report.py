"""Aggregate objects, not seeds; all zero-operation cases stay in the cohort."""
import argparse
from collections import Counter

import numpy as np

from .config import ARMS, OUT, PARAMS, folder, read, write, now


def get(record, keys):
    for key in keys.split('.'):
        if not isinstance(record, dict): return None
        record = record.get(key)
    return record


def object_summary(values):
    """values are (object index, seed, value); average seeds before bootstrap."""
    grouped = {}
    for index, seed, value in values:
        if value is None or not np.isfinite(value): continue
        grouped.setdefault(index, []).append(float(value))
    if not grouped: return dict(mean=None, ci95=None, objects=0, cases=0)
    means = np.array([np.mean(grouped[index]) for index in sorted(grouped)])
    rng = np.random.default_rng(PARAMS['random_seed'])
    boot = means[rng.integers(len(means), size=(5000, len(means)))].mean(1)
    return dict(mean=float(means.mean()), median_object=float(np.median(means)),
        ci95=np.quantile(boot, [.025, .975]).tolist(),
        objects=len(means), cases=sum(map(len, grouped.values())),
        improved_objects=int((means > 1.e-12).sum()), worsened_objects=int((means < -1.e-12).sum()),
        values_by_object={str(index): float(np.mean(grouped[index])) for index in sorted(grouped)})


def gain(record, endpoint, metric):
    before, after = get(record['original_metrics'], metric), get(record['metrics'][endpoint], metric)
    if before is None or after is None: return None
    return after - before if 'fscore' in metric else before - after


def aggregate(records, subset):
    metrics = ('surface.chamfer_l1', 'surface.fscore_0p02',
        'input.local_target_depth_mae', 'novel.local_target_depth_mae',
        'input.local_union_depth_mae', 'novel.local_union_depth_mae',
        'novel.global_target_depth_mae', 'novel.local_bad_depth_mae')
    output = {}
    for arm in ARMS:
        rows = [r for r in records if r['arm'] == arm and subset(r)]
        endpoints = {}
        for endpoint in ('proposal', 'selected', 'accepted'):
            item = {metric: object_summary([(r['index'], r['seed'], gain(r, endpoint, metric)) for r in rows]) for metric in metrics}
            for domain in ('input', 'novel'):
                key = f'{domain}.local_target_depth_mae'
                ratios = []
                for r in rows:
                    before, after = get(r['original_metrics'], key), get(r['metrics'][endpoint], key)
                    value = (before - after) / before if before is not None and before > 1.e-8 and after is not None else None
                    ratios.append((r['index'], r['seed'], value))
                item[f'{domain}.local_depth_relative_reduction'] = object_summary(ratios)
            endpoints[endpoint] = item
        durations = [r['elapsed_seconds'] for r in rows if r['steps'] > 0]
        output[arm] = dict(cases=len(rows), states=dict(Counter(r['status'] for r in rows)),
            selected_edits=sum(r.get('actual_selected_edit', False) for r in rows),
            accepted_edits=sum(r['input_acceptance']['accepted'] for r in rows),
            active_optimization_count=len(durations), active_mean_seconds=float(np.mean(durations)) if durations else None,
            max_peak_gib=max((r['peak_allocated_bytes'] / 2 ** 30 for r in rows), default=0),
            selected_more_components=sum(r['metrics']['selected']['structure']['component_count'] > r['original_metrics']['structure']['component_count'] for r in rows),
            selected_more_nonmanifold_edges=sum(r['metrics']['selected']['structure']['nonmanifold_edge_count'] > r['original_metrics']['structure']['nonmanifold_edge_count'] for r in rows),
            endpoints=endpoints)
    return output


def paired(records, left, right, predicate):
    indexed = {(r['index'], r['seed'], r['arm']): r for r in records}
    output = {}
    for metric in ('surface.chamfer_l1', 'input.local_target_depth_mae', 'novel.local_target_depth_mae'):
        values = []
        for index in PARAMS['indices']:
            for seed in PARAMS['seeds']:
                a, b = indexed.get((index, seed, left)), indexed.get((index, seed, right))
                if a is None or b is None or not predicate(a): continue
                x, y = gain(a, 'selected', metric), gain(b, 'selected', metric)
                values.append((index, seed, x - y if x is not None and y is not None else None))
        output[metric] = object_summary(values)
    return output


def fmt(value, digits=6):
    return 'NA' if value is None else f'{value:+.{digits}f}'


def build(final=False):
    records, missing = [], []
    expected = len(PARAMS['indices']) * len(PARAMS['seeds']) * len(ARMS)
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            for arm in ARMS:
                p = folder(index, seed) / arm / 'result.json'
                if not p.exists(): missing.append(str(p)); continue
                record = read(p)
                if record.get('status') == 'failed': missing.append(str(p)); continue
                records.append(record)
    if final and missing: raise ValueError(f'cannot finalize: {len(missing)} missing states')
    all_cases = aggregate(records, lambda r: True)
    active = aggregate(records, lambda r: r['target_diagnostic']['query_count'] > 0)
    pure_pair = aggregate(records, lambda r: r['target_diagnostic']['pure_paired_depth_comparison_valid'])
    pairwise = dict(
        extra_normals=paired(records, 'oracle_dense_normal', 'oracle_dense_depth', lambda r: True),
        depth_density=paired(records, 'oracle_dense_depth', 'oracle_sparse', lambda r: r['target_diagnostic']['query_count'] > 0),
        depth_accuracy_pure_pair=paired(records, 'oracle_sparse', 'input_sparse',
            lambda r: r['target_diagnostic']['pure_paired_depth_comparison_valid']))
    diagnostics = [{key: r[key] for key in ('index', 'seed', 'uid', 'target_diagnostic')}
        for r in records if r['arm'] == 'input_sparse']
    value = dict(time=now(), complete=not missing, expected_states=expected, states=len(records), missing=missing,
        all_cases=all_cases, active_input_cases=active, pure_paired_cases=pure_pair,
        pairwise=pairwise, target_diagnostics=diagnostics,
        parameters=PARAMS, records=records)
    write(OUT / 'summary.json', value)
    text = ['# 第六轮：真实几何错误与输入约束对照', '', f'时间：{value["time"]}', '',
        f'结果：{len(records)}/{expected}，状态：{"完整" if not missing else "未完成"}。', '',
        '同一批8物体×2 seeds；冻结30K权重，仅优化原SS上的局部latent。80步、更新上限0.5个训练标准差。',
        'A/B为使用GT的诊断；C为实际输入方法。A-depth辅助组用于区分法线与深度密度。', '',
        '所有改善以正值为好。各物体先平均seed，再平均物体；区间为5000次物体bootstrap。', '']
    labels = dict(oracle_dense_normal='A：密集GT深度+法线', oracle_dense_depth='A-depth：密集GT深度',
        oracle_sparse='B：相同查询处GT深度', input_sparse='C：输入估计深度')
    for title, group in (('全体案例', all_cases), ('有输入匹配的案例', active), ('B/C全部查询均有GT交点的严格配对子集', pure_pair)):
        text += [f'## {title}', '', '|组别|案例|选出修改/输入门接受|原输入局部深度下降|新视角局部深度下降|CD改善|F@.02改善|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in ARMS:
            arm = group[name]; e = arm['endpoints']['selected']
            input_reduction = e['input.local_depth_relative_reduction']['mean']
            novel_reduction = e['novel.local_depth_relative_reduction']['mean']
            text.append(f'|{labels[name]}|{arm["cases"]}|{arm["selected_edits"]}/{arm["accepted_edits"]}|'
                f'{"NA" if input_reduction is None else f"{input_reduction:+.2%}"}|'
                f'{"NA" if novel_reduction is None else f"{novel_reduction:+.2%}"}|'
                f'{fmt(e["surface.chamfer_l1"]["mean"])}|{fmt(e["surface.fscore_0p02"]["mean"],4)}|')
        text += ['']
    text += ['## 配对差异', '', '正值表示左侧监督组比右侧改善更多。', '',
        '|对比|CD改善差及95%区间|新视角局部深度改善差及95%区间|有效案例/物体|', '|---|---:|---:|---:|']
    for name, pair in pairwise.items():
        cd, depth = pair['surface.chamfer_l1'], pair['novel.local_target_depth_mae']
        text.append(f'|{name}|{fmt(cd["mean"])} {cd["ci95"]}|{fmt(depth["mean"])} {depth["ci95"]}|{cd["cases"]}/{cd["objects"]}|')
    indexed = {(r['index'], r['seed'], r['arm']): r for r in records}
    text += ['', '## 约束质量与缺失', '', '|物体/seed|密集GT训练射线|输入查询|有GT交点|原深度MAE|输入目标MAE|正确方向比例|', '|---|---:|---:|---:|---:|---:|---:|']
    for record in diagnostics:
        d = record['target_diagnostic']
        dense_count = indexed.get((record['index'], record['seed'], 'oracle_dense_depth'), {}).get('query_count')
        text.append(f'|{record["index"]:03d}/{record["seed"]}|{dense_count}|{d["query_count"]}|{d["gt_intersections"]}|'
            f'{fmt(d["original_depth_mae_at_queries"])}|{fmt(d["input_estimate_depth_mae"])}|{d["correct_direction_fraction"]}|')
    text += ['', '## 耗时与结构', '', '|组别|有效优化案例|平均优化秒数|PyTorch峰值allocated GiB|选出结果增加分量/非流形边|', '|---|---:|---:|---:|---:|']
    for name, arm in all_cases.items():
        text.append(f'|{labels[name]}|{arm["active_optimization_count"]}|{arm["active_mean_seconds"]}|{arm["max_peak_gib"]:.2f}|'
            f'{arm["selected_more_components"]}/{arm["selected_more_nonmanifold_edges"]}|')
    text += ['', '## 解释边界', '',
        '- 主表是各组自身训练损失选出的结果；固定80步proposal、输入门接受后的结果另见JSON。Oracle选出结果不能作为可部署方法成绩。',
        '- 输入门只检查训练目标选出的candidate，不重新搜索全部checkpoint；不能与第五轮按输入门逐checkpoint选择的接受率直接比较。',
        '- 轮廓门沿用与原mesh重叠至少0.995的条件，并非相对观测mask的改善门；它可能拒绝正确的轮廓纠错。selected候选和accepted结果必须分别解释。',
        '- GT不用于C的区域、目标、可更新行、checkpoint选择或接受判断。C在本案例访问GT前完成；进程可能驻留上一seed的GT缓存，驻留不等于输入分支访问。',
        '- B保持相同查询像素；无GT交点时不伪造正深度。仅严格配对子集可将B/C差异解释为深度准确性影响。',
        '- input项包含4个原输入视角，未等同于训练视角；新8视角完全不用于优化或选择，但物体已在开发中检查过。',
        '- region来自原mesh的输入错误区域；新视角按同一批原mesh面片投影，不能据此证明任意隐藏缺失结构修复。',
        '- 没有重新对齐、ICP或按组归一化。局部误差分母固定，消失射线付0.1。',
        '- 当前有限预算的oracle结果是编辑能力诊断，不是数学上的全局最优上限。',
        '- 近邻顶点位移按原mesh顶点面积加权，是改动面积近似，不是精确点到三角面距离。',
        '- 表面CD/F-score由固定随机种子的50000点采样估计；极小差异仍可能受采样波动影响，需结合固定相机深度误差判断。',
        '- 优化计时含候选保存，不含模型加载、最终指标和渲染；无操作的零耗时不计入有效优化平均。',
        '- 仅使用已有扫描物体的渲染输入，不是新增真实拍摄或独立物体测试。', '',
        f'缺失结果：{len(missing)}。完整逐例数据：`{OUT / "summary.json"}`。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(text))
    print(dict(complete=not missing, states=len(records), expected=expected, report=str(OUT / 'REPORT.md')), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--final', action='store_true')
    args = parser.parse_args(); build(args.final)


if __name__ == '__main__': main()
