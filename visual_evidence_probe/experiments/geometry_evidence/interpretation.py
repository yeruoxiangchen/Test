"""Evidence-based closeout prose and explicit acceptance-gate limitations."""
import numpy as np

from .config import ARMS, PARAMS, folder, read, write, OUT
from .report import object_summary


def percent(value):
    return 'NA' if value is None else f'{value:+.2%}'


def main(summary):
    records = summary['records']
    states = summary['all_cases']
    def endpoint(arm): return states[arm]['endpoints']['selected']
    def reduction(arm, domain='novel'):
        return endpoint(arm)[f'{domain}.local_depth_relative_reduction']
    active = [r for r in records if r['arm'] == 'input_sparse' and r['target_diagnostic']['query_count'] > 0]
    pure = [r for r in active if r['target_diagnostic']['pure_paired_depth_comparison_valid']]
    observed = {arm: [] for arm in ARMS}
    gate_counts = {arm: dict(improved_observed_mask_but_rejected=0, evaluated=0) for arm in ARMS}
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            root = folder(index, seed)
            d = read(root / 'observed_silhouette_diagnostic.json')
            for arm in ARMS:
                row = d['rows'][str(root / arm / 'selected.obj')]
                observed[arm].append((index, seed, row['mean_iou_gain_to_observed']))
                gate_counts[arm]['evaluated'] += 1
                gate_counts[arm]['improved_observed_mask_but_rejected'] += int(
                    row['mean_iou_gain_to_observed'] > 1.e-6 and not row['accepted_by_legacy_gate'])
    gate_diagnostic = dict(observed_iou_gain={arm: object_summary(values) for arm, values in observed.items()},
        counts=gate_counts, posthoc_only=True, changes_selection=False)
    write(OUT / 'gate_diagnostic_summary.json', gate_diagnostic)
    text = ['# 第六轮收尾结论', '', f'时间：{summary["time"]}', '',
        '**真实重建误差上的四组对照已经完成：8物体×2 seeds×4组，共64条结果。**', '',
        '这轮直接纠正原30K输出与扫描几何之间的误差，使用相同的原SS、局部可更新行和优化预算。',
        'A/B使用GT，是定位瓶颈的诊断；C才是输入图像提供纠错依据的分支。', '',
        '|候选组别|原输入局部深度误差下降|新视角局部深度误差下降|全局CD改善|新视角下降95%区间|',
        '|---|---:|---:|---:|---:|']
    labels = dict(oracle_dense_normal='A：密集正确深度+法线', oracle_dense_depth='A-depth：密集正确深度',
        oracle_sparse='B：原匹配位置的正确深度', input_sparse='C：原输入估计深度')
    for arm in ARMS:
        r = reduction(arm)
        text.append(f'|{labels[arm]}|{percent(reduction(arm,"input")["mean"])}|{percent(r["mean"])}|'
            f'{endpoint(arm)["surface.chamfer_l1"]["mean"]:+.6f}|'
            f'[{percent(r["ci95"][0])}, {percent(r["ci95"][1])}]|')
    text += ['', '## 能得出的判断', '',
        f'1. **真实误差下的编辑能力。** A-depth在新视角的平均局部误差下降为{percent(reduction("oracle_dense_depth")["mean"])}，'
        f'8个物体中有{reduction("oracle_dense_depth")["improved_objects"]}个按seed平均后改善。'
        '这比第五轮的微小人工latent扰动更接近实际纠错，但仍是当前预算下的有监督诊断。',
        f'2. **输入证据的覆盖。** 16个案例中{len(active)}个有输入匹配，另{16-len(active)}个对B/C保留原结果。'
        'A-depth与B的配对差异用于检查约束密度及其空间/视图覆盖，不能把所有差异都归结为点数。',
        f'3. **输入证据的准确性。** B/C有{len(pure)}个案例、来自{len({r["index"] for r in pure})}个物体，满足全部查询均与GT相交。'
        '判断深度准确性的影响应看严格配对子集；没有GT交点的查询另计，C没有据此筛点。',
        '4. **结构稳定性。** 降低深度误差并不保证连通结构完好。各组增加分量与非流形边的案例已单独统计。', '',
        '## 候选与接受结果', '',
        f'C选出修改{states["input_sparse"]["selected_edits"]}/16，输入门最终接受{states["input_sparse"]["accepted_edits"]}/16。'
        '旧门约束相对原mesh的轮廓变化，而且只检查本轮训练损失选中的候选；它可能拒绝有益的轮廓纠错。'
        '因此不能把接受率直接当作编辑能力，也不能与第五轮的逐checkpoint输入门搜索做直接配对。', '',
        f'**最终接受输出尚未形成精度提升。** C接受后的新视角局部误差下降为'
        f'{percent(states["input_sparse"]["endpoints"]["accepted"]["novel.local_depth_relative_reduction"]["mean"])}，'
        f'全局CD改善为{states["input_sparse"]["endpoints"]["accepted"]["surface.chamfer_l1"]["mean"]:+.8f}。'
        '前表的4组数值是纠错候选，不能替代实际接受输出的结果。', '',
        '|组别|选出候选增加分量|选出候选增加非流形边|观测mask平均IoU改善|观测mask改善但仍被旧门拒绝|',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        text.append(f'|{labels[arm]}|{states[arm]["selected_more_components"]}/16|{states[arm]["selected_more_nonmanifold_edges"]}/16|'
            f'{gate_diagnostic["observed_iou_gain"][arm]["mean"]:+.6f}|{gate_counts[arm]["improved_observed_mask_but_rejected"]}/16|')
    text += ['', '新增分量的数量不等于整体形状崩溃；逐例最大分量顶点占比应与原30K对照，很多新增分量很小。', '']
    a, c = reduction('oracle_dense_depth')['mean'], reduction('input_sparse')['mean']
    if a is not None and a > 0 and c is not None and a > c:
        recommendation = ('下一步优先提高输入几何约束的空间与视图覆盖，同时抑制局部编辑产生的碎片和区域外变化。'
            '先用稠密正确几何的收益作为目标，检验新的输入几何估计能否接近它，再决定训练哪一部分。'
            '本轮没有证明简单增加若干同类匹配点就足够，也没有建立独立真实拍摄泛化结论。')
    else:
        recommendation = ('当前有监督编辑在新视角上的优势仍不足，应继续检查可更新范围、目标可达性和区域外保护之间的冲突，'
            '并同时审计输入几何约束；尚不适合将瓶颈单独归因于输入估计。')
    text += ['', '## 后续方向', '', recommendation, '',
        '## 验证与文件', '',
        '- 8项协议与评价验证通过；全部SS坐标、未更新latent行及拒绝后的原latent恢复已逐文件验证。',
        '- 49张图包括16组mesh对比、16组深度误差图、16组输入区域/匹配点图及1张汇总图。',
        '- 所有mesh、图像、latent、缓存和日志存于`/data/zjr`。30K模型权重未训练。',
        '- 8个新视角未用于本轮优化和选择；物体仍是已检查的开发集，输入来自扫描物体渲染。',
        f'- 完整结果：`{OUT}`。', '', '---', '']
    return '\n'.join(text), recommendation
