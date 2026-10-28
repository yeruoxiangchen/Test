"""Aggregate complete paired results, including rejections and diagnostic oracles."""
import numpy as np
from PIL import Image, ImageDraw

from .config import BASE, OUT, INDICES, SEEDS, folder, observations, read, write, verify, now
from .policy import THRESHOLDS


def summarize(records, baseline, accepted=None):
    values = {}
    for metric, field in [('cd', 'chamfer_l1'), ('f02', 'fscore_0p02')]:
        by_object = [float(np.mean([r['surface'][field] for r in records if r['index'] == i])) for i in INDICES]
        ref = [float(np.mean([r['surface'][field] for r in baseline if r['index'] == i])) for i in INDICES]
        if not np.isfinite(by_object).all():
            raise ValueError('missing object in summary')
        gain = (np.asarray(ref) - by_object) * (1 if metric == 'cd' else -1)
        values[metric] = dict(mean=float(np.mean(by_object)), per_object=by_object,
                             gain_over_real2=float(np.mean(gain)), wins=int((gain > 1.e-9).sum()),
                             losses=int((gain < -1.e-9).sum()), ties=int((np.abs(gain) <= 1.e-9).sum()))
    occupancy = [r['occupancy']['iou'] for r in records if r.get('occupancy') is not None]
    values.update(records=len(records), ss_iou=float(np.mean(occupancy)) if len(occupancy) == len(records) else None,
                  accepted_objects=accepted, acceptance_rate=None if accepted is None else len(accepted)/len(INDICES))
    return values


def gallery(index, old, context):
    import trimesh
    from visual_evidence_probe.render import render
    rec, k, w, _ = observations(index)
    selected = next(r['selected'] for r in read(OUT / 'raw_decisions.json')['records'] if r['index'] == index)
    target = next(r for r in read(OUT / 'evaluation_targets.json')['records'] if r['index'] == index)
    rows = [('GT', verify(target['gt']), None)]
    for seed in SEEDS:
        record = old[index, 'real2', seed]
        rows.append((f'real2 / {seed}', verify(record['mesh']), record))
    for key in ('a', 'b'):
        record = read(folder(index) / 'refined' / f'{key}.metrics.json')
        rows.append((f'direct similarity {key}', verify(record['mesh']), record))
    for seed in SEEDS:
        record = old[index, f'donor_{selected}6', seed]
        rows.append((f'raw {selected} + six / {seed}', verify(record['mesh']), record))
    for seed in SEEDS:
        record = read(folder(index) / 'resampled' / f'{seed}.metrics.json')
        rows.append((f'aligned {selected} + six / {seed}', verify(record['mesh']), record))
    side, label = 230, 220
    canvas = Image.new('RGB', (label + side*4, side*len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, path, record) in enumerate(rows):
        draw.text((8, row*side+55), name, fill='black')
        if record:
            draw.text((8, row*side+80), f"CD {record['surface']['chamfer_l1']:.5f}", fill='black')
            draw.text((8, row*side+100), f"F@.02 {record['surface']['fscore_0p02']:.4f}", fill='black')
        mesh = trimesh.load(path, force='mesh', process=False)
        images, _ = render(mesh, k[[0, 1, 3, 6]], w[[0, 1, 3, 6]], side=side, context=context)
        for j, image in enumerate(images):
            canvas.paste(image, (label+j*side, row*side))
    canvas.save(folder(index) / 'mesh_comparison.jpg', quality=93)


def report(context=None):
    old = {}
    for index in INDICES:
        for arm in ('real2', 'repeat8', 'real8', 'donor_a6', 'donor_b6', 'mixed_ab6', 'mixed_ba6'):
            for seed in SEEDS:
                record = read(BASE / 'objects' / f'{index:03d}' / 'results' / arm / f'{seed}.json')
                if not record['success'] or (record['index'], record['seed'], record['arm']) != (index, seed, arm):
                    raise ValueError('invalid existing result')
                verify(record['mesh'])
                old[index, arm, seed] = record
    baseline = [old[i, 'real2', s] for i in INDICES for s in SEEDS]
    raw = read(OUT / 'raw_decisions.json')['records']
    aligned = read(OUT / 'aligned_decisions.json')['records']
    summaries = {}
    for arm in ('real2', 'repeat8', 'real8'):
        summaries[arm] = summarize([old[i, arm, s] for i in INDICES for s in SEEDS], baseline)
    summaries['always_add_random_source'] = summarize(
        [old[i, f'donor_{k}6', s] for i in INDICES for k in ('a', 'b') for s in SEEDS], baseline, list(INDICES))
    summaries['raw_best_source'] = summarize(
        [old[r['index'], f"donor_{r['selected']}6", s] for r in raw for s in SEEDS], baseline, list(INDICES))
    for threshold in THRESHOLDS:
        records, accepted = [], []
        for row in raw:
            key = row['gated'][str(threshold)]
            if key is not None:
                accepted.append(row['index'])
            arm = f'donor_{key}6' if key else 'real2'
            records.extend(old[row['index'], arm, s] for s in SEEDS)
        summaries[f'raw_gate_{threshold:.2f}'] = summarize(records, baseline, accepted)
    direct = [read(folder(i) / 'refined' / f'{k}.metrics.json') for i in INDICES for k in ('a', 'b')]
    generated = {(i, s): read(folder(i) / 'resampled' / f'{s}.metrics.json') for i in INDICES for s in SEEDS}
    for record in direct + list(generated.values()):
        if not record['success']:
            raise ValueError('new mesh failure; cannot use success-only summary')
        verify(record['mesh'])
    summaries['direct_similarity'] = summarize(direct, baseline)
    summaries['aligned_source_add'] = summarize(list(generated.values()), baseline, list(INDICES))
    for threshold in THRESHOLDS:
        records, accepted = [], []
        for row in aligned:
            keep = row['gated'][str(threshold)] is not None
            if keep:
                accepted.append(row['index'])
            records.extend(generated[row['index'], s] if keep else old[row['index'], 'real2', s] for s in SEEDS)
        summaries[f'aligned_gate_{threshold:.2f}'] = summarize(records, baseline, accepted)
    pool = ('real2', 'donor_a6', 'donor_b6', 'mixed_ab6', 'mixed_ba6')
    object_oracle, seed_oracle, oracle_choices = [], [], []
    for i in INDICES:
        chosen = min(pool, key=lambda a: np.mean([old[i, a, s]['surface']['chamfer_l1'] for s in SEEDS]))
        object_oracle.extend(old[i, chosen, s] for s in SEEDS)
        seed_choices = {}
        for s in SEEDS:
            choice = min(pool, key=lambda a: old[i, a, s]['surface']['chamfer_l1'])
            seed_oracle.append(old[i, choice, s])
            seed_choices[str(s)] = choice
        oracle_choices.append(dict(index=i, object_choice=chosen, per_seed_choice=seed_choices))
    summaries['gt_oracle_object_CD_only'] = summarize(object_oracle, baseline)
    summaries['gt_oracle_per_seed_CD_only'] = summarize(seed_oracle, baseline)
    fits = [read(folder(i) / 'refined' / f'{k}.json') for i in INDICES for k in ('a', 'b')]
    quality = dict(before_mean=float(np.mean([r['initial']['ious'] for r in fits])),
                   after_mean=float(np.mean([r['chosen']['ious'] for r in fits])),
                   before_per_object=[float(np.mean([r['initial']['ious'] for r in fits if r['index'] == i])) for i in INDICES],
                   after_per_object=[float(np.mean([r['chosen']['ious'] for r in fits if r['index'] == i])) for i in INDICES],
                   fit_seconds_total=float(sum(r['elapsed_seconds'] for r in fits)),
                   fit_seconds_mean=float(np.mean([r['elapsed_seconds'] for r in fits])),
                   identity_fallbacks=sum(r['chosen']['step'] == 0 for r in fits))
    result = dict(time=now(), objects=4, new_resampled_meshes=8, refined_meshes=8,
        scope='same four previously inspected objects, one rendered HSSD scene; exploratory',
        summaries=summaries, raw_decisions=raw, aligned_decisions=aligned, oracle_choices=oracle_choices,
        silhouette_quality=quality, no_gt_for_selection_or_fitting=True, all_results_successful=True,
        oracle_is_privileged=True, fit_is_not_gt_alignment=True,
        cost_note='Selection requires two donor reconstructions. Refinement fits both; re-sampling adds SS25+SLat25. Gates reuse existing branches offline, not independent new runs.')
    write(OUT / 'summary.json', result)
    lines = ['# 第二轮：输入一致性与简单全局修正', '', f"完成时间：{result['time']}", '',
        '四个此前已检查的 HSSD 对象、两个配对采样seed。阈值网格事后固定，不能视为独立测试。',
        '全部对象和回退结果均计入；接受率是四个对象中的比例。real8使用额外真实观测，作为参考。', '',
        '|方案|CD↓|F@.02↑|接受生成来源|CD优于/差于/等于real2的对象数|',
        '|---|---:|---:|---:|---:|']
    for name, values in summaries.items():
        cd, f = values['cd'], values['f02']
        accepted = '-' if values['accepted_objects'] is None else f"{len(values['accepted_objects'])}/4"
        lines.append(f"|{name}|{cd['mean']:.8f}|{f['mean']:.6f}|{accepted}|{cd['wins']}/{cd['losses']}/{cd['ties']}|")
    lines += ['', '## 轮廓与三维指标的关系', '',
        f"直接相似变换：两图轮廓IoU均值 {quality['before_mean']:.6f} → {quality['after_mean']:.6f}。",
        f"每个供图mesh拟合平均 {quality['fit_seconds_mean']:.2f} 秒，共 {quality['fit_seconds_total']:.2f} 秒。",
        '直接修正保持局部结构和拓扑，只调整旋转、平移、统一尺度；不报告旧SS占据指标。',
        '生成分支使用原始轮廓预选的相同来源（A/B/A/A），修正前后只改变全局几何对齐。',
        '轮廓变好不等于不可见部分的几何变好，需要共同查看CD、F与渲染图。', '',
        '## 选择器的诊断天花板', '',
        'Oracle池只含real2和四个生成供图分支，不含real8或GT渲染。它用GT选择最低CD，',
        '仅回答“现有候选是否有可选的更好结果”，不能作为可部署方法，也不能据此选择门槛。',
        '逐seed oracle更加乐观；表中F是最低CD分支对应的F，并非独立最大化F。', '',
        '## 代价与复现', '',
        result['cost_note'], '',
        '```bash', 'bash visual_evidence_probe/run_consistency.sh prepare',
        'EVIDENCE_GPU=4 bash visual_evidence_probe/run_consistency.sh run', '```', '',
        '[完整数据](summary.json)；[固定方案](../../experiments/input_consistency/PROTOCOL.md)。', '']
    for i in INDICES:
        lines.append(f'- 对象{i:03d}：[全部mesh对照](objects/{i:03d}/mesh_comparison.jpg)；[修正OBJ](objects/{i:03d}/refined/)；[再次生成OBJ](objects/{i:03d}/resampled/)。')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if context is not None:
        for i in INDICES:
            gallery(i, old, context)
    print(OUT / 'REPORT.md', flush=True)
    return result
