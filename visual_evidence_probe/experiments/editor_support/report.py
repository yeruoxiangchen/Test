"""Small-JSON-only reporting for the fifth, exploratory diagnostic round.

Default invocation writes a partial report and never appends the audit.  Use
``--final`` only after all expected arms have finished; incomplete final reports
are still written, return exit status 2, and leave the audit untouched.
No mesh, checkpoint, latent, image, or scan is loaded by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import OUT, STORE, PARAMS, now, write


INDICES = tuple(range(4, 12))
RECOVERY_INDICES = (4, 6)
SEEDS = (42, 43)
SUPPORTS = ('original', 'expanded')
OBJECTIVES = ('proxy', 'mesh')
ENDPOINTS = ('step0', 'proposal', 'accepted')
METRICS = ('chamfer_l1', 'fscore_0p02')
BOOTSTRAPS = 5000
FACTORIAL_TERMINAL = {'complete', 'init_failure', 'unable_validation', 'no_input_constraints'}
VALID_INITIALIZATION = {'valid', 'no_support_change', 'no_input_constraints'}
LIMITATIONS = [
    '没有新增独立数据：仍是已检查过的8个scan物体、每物体2个seed；输入均为这些scan物体的渲染视图。',
    '四个known-latent案例是decoder可表达目标上的开发诊断；mesh至少3/4达到50%深度误差下降仅是development gate，不是泛化结论。',
    'scan GT仅用于最终评价，不用于目标构造、梯度、SS扩展、初始化或checkpoint选择；本报告只读取已保存JSON，不能独立重验该数据流。',
    'holdout仅从本轮edit方向估计/优化排除；原生成、全局对齐及先前区域选择已见全部4视图，因此不是独立测试视角。',
    '扩展SS与新增token保形初始化是联合干预；init_failure保留，不可将初始化失败归因于SS表达能力。coarsecell存在数不等于decoder角点覆盖率。',
    '全部案例与valid-init子集分别统计；no_input_constraints作为原mesh无操作保留。缺失指标不补零，partial均值不能代表完整8物体实验。',
    '先在每个物体内平均该子集可用seed，再平均物体；95%区间来自5000次物体bootstrap，不将16个seed当独立物体。子集可能每物体只有1个有效seed，JSON列明数量。',
    '计时使用optimization elapsed_seconds：包含snapshot和候选保存，不含GT评价；不以case complete总耗时替代。相加是各任务耗时和，不是并行作业总墙钟时间。',
    '原生spconv自定义CUDA backward尚无跨进程bitwise确定性认证；硬件仅引用已有JSON metadata，缺失时不推断。',
]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _load(path):
    if not path.exists():
        return None, None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return None, 'JSON root must be an object'
        return value, None
    except (OSError, ValueError) as error:
        return None, f'{type(error).__name__}: {error}'


def _surface(metrics, key):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get('surface', {}).get(key)
    return float(value) if _finite(value) else None


def _metric_errors(record):
    errors = []
    for baseline, value in [('original_metrics', record.get('original_metrics'))] + [
            (endpoint, record.get('metrics', {}).get(endpoint)) for endpoint in ENDPOINTS]:
        for key in METRICS:
            if _surface(value, key) is None:
                errors.append(f'{baseline}.surface.{key}')
    for key in ('original_structure', 'proposal_structure'):
        value = record.get(key)
        if not isinstance(value, dict) or not all(_finite(value.get(k)) for k in
                ('component_count', 'nonmanifold_edge_count', 'boundary_edge_count')):
            errors.append(key)
    if not isinstance(record.get('accepted_edit'), bool):
        errors.append('accepted_edit')
    return errors


def _optimization(path, result):
    optimization, error = _load(path / 'optimization.json')
    source = optimization if optimization is not None else (result or {})
    elapsed = source.get('elapsed_seconds')
    if result and result.get('status') == 'no_input_constraints':
        elapsed = 0.0
    return dict(elapsed_seconds=float(elapsed) if _finite(elapsed) else None,
        peak_allocated_bytes=source.get('peak_allocated_bytes'),
        source=str(path / ('optimization.json' if optimization is not None else 'result.json')),
        optimization_json_error=error,
        includes_snapshots=True, includes_gt_evaluation=False)


def _object_stats(values, expected_cases):
    """Average seeds inside each object, then bootstrap objects, never seeds."""
    grouped = {}
    for index, seed, value in values:
        grouped.setdefault(int(index), {})[int(seed)] = float(value)
    object_values = {str(i): float(np.mean(list(rows.values()))) for i, rows in sorted(grouped.items())}
    array = np.asarray(list(object_values.values()), dtype=np.float64)
    if len(array) >= 2:
        rng = np.random.default_rng(PARAMS['random_seed'])
        boot = array[rng.integers(len(array), size=(BOOTSTRAPS, len(array)))].mean(axis=1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    else:
        interval = None
    return dict(mean=float(array.mean()) if len(array) else None,
        object_bootstrap_ci95=interval, bootstrap_resamples=BOOTSTRAPS,
        object_count=len(array), case_count=len(values), expected_case_count=expected_cases,
        complete_case_set=len(values) == expected_cases,
        object_values=object_values,
        seed_counts_by_object={str(i): len(rows) for i, rows in sorted(grouped.items())},
        by_seed={str(seed): float(np.mean([rows[seed] for rows in grouped.values() if seed in rows]))
                 for seed in SEEDS if any(seed in rows for rows in grouped.values())},
        win_objects=int((array > 1.e-8).sum()), loss_objects=int((array < -1.e-8).sum()),
        tie_objects=int((np.abs(array) <= 1.e-8).sum()), positive_means_better=True)


def _gain(record, endpoint, baseline, metric):
    base = record.get('original_metrics') if baseline == 'original' else record.get('metrics', {}).get('step0')
    a, b = _surface(base, metric), _surface(record.get('metrics', {}).get(endpoint), metric)
    if a is None or b is None:
        return None
    return a - b if metric == 'chamfer_l1' else b - a


def _timing(records):
    values = [r['timing']['elapsed_seconds'] for r in records if _finite(r['timing']['elapsed_seconds'])]
    peaks = [r['timing']['peak_allocated_bytes'] for r in records if _finite(r['timing']['peak_allocated_bytes'])]
    return dict(recorded_cases=len(values), sum_optimization_seconds=float(sum(values)),
        mean_optimization_seconds=float(np.mean(values)) if values else None,
        max_peak_allocated_bytes=max(peaks) if peaks else None,
        includes_snapshots=True, includes_gt_evaluation=False)


def _group(records, keys):
    rows = [r for r in records if (r['index'], r['seed']) in keys]
    metrics = {}
    for baseline in ('original', 'step0'):
        metrics[baseline] = {}
        for endpoint in ENDPOINTS:
            metrics[baseline][endpoint] = {}
            for metric in METRICS:
                values = [(r['index'], r['seed'], value) for r in rows
                          if (value := _gain(r['result'], endpoint, baseline, metric)) is not None]
                metrics[baseline][endpoint][metric] = _object_stats(values, len(keys))
    return dict(expected_cases=len(keys), found_results=sum(r['has_result'] for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows)),
        accepted_edits=sum(r['result'].get('accepted_edit') is True for r in rows),
        no_input_cases=sum(r['status'] == 'no_input_constraints' for r in rows),
        initialization_status_counts=dict(Counter(r['initialization_status'] for r in rows)),
        improvements=metrics, timing=_timing(rows))


def _contrasts(records, keys):
    by_key = {(r['index'], r['seed'], r['support'], r['objective']): r['result'] for r in records}
    definitions = {
        'mesh_minus_proxy_original': [('original', 'mesh', 1), ('original', 'proxy', -1)],
        'mesh_minus_proxy_expanded': [('expanded', 'mesh', 1), ('expanded', 'proxy', -1)],
        'expanded_minus_original_proxy': [('expanded', 'proxy', 1), ('original', 'proxy', -1)],
        'expanded_minus_original_mesh': [('expanded', 'mesh', 1), ('original', 'mesh', -1)],
        'interaction': [('expanded', 'mesh', 1), ('expanded', 'proxy', -1),
                        ('original', 'mesh', -1), ('original', 'proxy', 1)],
    }
    output = {}
    for name, terms in definitions.items():
        output[name] = {}
        for baseline in ('original', 'step0'):
            output[name][baseline] = {}
            for endpoint in ('proposal', 'accepted'):
                output[name][baseline][endpoint] = {}
                for metric in METRICS:
                    values = []
                    for index, seed in sorted(keys):
                        gains = [_gain(by_key.get((index, seed, support, objective), {}), endpoint, baseline, metric)
                                 for support, objective, _ in terms]
                        if all(value is not None for value in gains):
                            values.append((index, seed, sum(value * term[2] for value, term in zip(gains, terms))))
                    output[name][baseline][endpoint][metric] = _object_stats(values, len(keys))
    return output


def _structure(records):
    output = []
    for r in records:
        value = r['result']
        original = value.get('original_structure')
        row = dict(index=r['index'], seed=r['seed'], support=r['support'], objective=r['objective'],
            original=original, proposal=value.get('proposal_structure'),
            accepted=value.get('accepted_structure'), candidate_selected=value.get('candidate_selected_structure'))
        for endpoint in ('proposal', 'accepted', 'candidate_selected'):
            if isinstance(original, dict) and isinstance(row[endpoint], dict):
                row[endpoint + '_deltas'] = {key: row[endpoint][key] - original[key]
                    for key in ('component_count', 'boundary_edge_count', 'nonmanifold_edge_count')
                    if _finite(original.get(key)) and _finite(row[endpoint].get(key))}
        output.append(row)
    return output


def summarize(out=OUT):
    """Read expected small JSON paths and return an auditable partial/full dict."""
    out = Path(out)
    missing, failures, schema_errors = [], [], []
    recovery = []
    for index in RECOVERY_INDICES:
        for seed in SEEDS:
            target, target_error = _load(out / 'recovery' / f'{index:03d}' / str(seed) / 'target.json')
            for objective in OBJECTIVES:
                path = out / 'recovery' / f'{index:03d}' / str(seed) / objective
                record, error = _load(path / 'result.json')
                key = dict(phase='recovery', index=index, seed=seed, objective=objective)
                status = record.get('status', 'unknown') if record else ('unreadable' if error else 'missing')
                if record is None:
                    (failures if error else missing).append(dict(**key, error=error, path=str(path / 'result.json')))
                elif status != 'complete':
                    failures.append(dict(**key, status=status, error=record.get('error')))
                else:
                    reduction = record.get('depth_error_reduction', record.get('depth_reduction'))
                    if not _finite(reduction) or not isinstance(record.get('recovery_passed'), bool):
                        schema_errors.append(dict(**key, fields=['depth_error_reduction', 'recovery_passed']))
                    elif record['recovery_passed'] != bool(reduction >= PARAMS['recovery_min_depth_reduction']):
                        schema_errors.append(dict(**key, fields=['recovery_passed disagrees with depth reduction']))
                recovery.append(dict(**key, path=str(path / 'result.json'), status=status,
                    has_result=record is not None, result=record or {}, target=target,
                    target_json_error=target_error, timing=_optimization(path, record)))
    recovery_groups = {}
    for objective in OBJECTIVES:
        rows = [r for r in recovery if r['objective'] == objective]
        values = [r['result'].get('depth_error_reduction', r['result'].get('depth_reduction')) for r in rows]
        ready = all(r['status'] == 'complete' and _finite(v) and isinstance(r['result'].get('recovery_passed'), bool)
                    for r, v in zip(rows, values))
        passes = sum(r['status'] == 'complete' and r['result'].get('recovery_passed') is True and
                     _finite(v) and v >= PARAMS['recovery_min_depth_reduction'] for r, v in zip(rows, values))
        recovery_groups[objective] = dict(expected_cases=4, complete_cases=sum(r['status'] == 'complete' for r in rows),
            passed_cases=passes, required_pass_cases=3, minimum_depth_reduction=.5,
            development_gate_passed=bool(passes >= 3) if ready else None,
            mean_depth_reduction=float(np.mean([v for v in values if _finite(v)])) if any(_finite(v) for v in values) else None,
            depth_reductions=[dict(index=r['index'], seed=r['seed'], value=v if _finite(v) else None,
                                  recovery_passed=r['result'].get('recovery_passed'), status=r['status'])
                              for r, v in zip(rows, values)], timing=_timing(rows))

    factorial, cases = [], []
    all_keys = {(index, seed) for index in INDICES for seed in SEEDS}
    for index, seed in sorted(all_keys):
        path = out / 'factorial' / f'{index:03d}' / str(seed)
        support, support_error = _load(path / 'expansion' / 'support.json')
        initialization, initialization_error = _load(path / 'expansion' / 'initialization.json')
        pool, pool_error = _load(path / 'input_pool.json')
        completion, completion_error = _load(path / 'complete.json')
        case_rows = []
        for support_name in SUPPORTS:
            for objective in OBJECTIVES:
                arm = path / f'{support_name}_{objective}'
                record, error = _load(arm / 'result.json')
                selected, selected_error = _load(arm / 'input_selected.json') if record is None else (None, None)
                key = dict(phase='factorial', index=index, seed=seed, support=support_name, objective=objective)
                status = record.get('status', 'unknown') if record else ('unreadable' if error else 'missing')
                if record is None:
                    (failures if error else missing).append(dict(**key, error=error,
                        path=str(arm / 'result.json'), input_selected_status=(selected or {}).get('status')))
                elif status not in FACTORIAL_TERMINAL:
                    failures.append(dict(**key, status=status, error=record.get('error')))
                if record is not None:
                    fields = _metric_errors(record)
                    if fields:
                        schema_errors.append(dict(**key, fields=fields))
                value = record or selected or {}
                init_status = value.get('initialization_status', 'original' if support_name == 'original' else
                    ('no_input_constraints' if status == 'no_input_constraints' else (initialization or {}).get('status', 'unknown')))
                row = dict(**key, path=str(arm / 'result.json'), status=status,
                    has_result=record is not None, result=record or {},
                    pending_input_selection=selected, input_selected_json_error=selected_error,
                    initialization_status=init_status, timing=_optimization(arm, record or selected))
                factorial.append(row); case_rows.append(row)
        expanded_statuses = {r['initialization_status'] for r in case_rows if r['support'] == 'expanded'}
        if len(expanded_statuses) == 1:
            init_status = next(iter(expanded_statuses))
        else:
            init_status = 'inconsistent'
            schema_errors.append(dict(phase='factorial', index=index, seed=seed,
                fields=['expanded objectives have different initialization_status']))
        cases.append(dict(index=index, seed=seed, input_pool=pool, expansion=support,
            initialization=initialization, initialization_status=init_status,
            valid_initialization=init_status in VALID_INITIALIZATION, case_completion=completion,
            json_errors={name: error for name, error in (('support', support_error),
                ('initialization', initialization_error), ('input_pool', pool_error), ('complete', completion_error)) if error}))
    valid_keys = {(r['index'], r['seed']) for r in cases if r['valid_initialization']}
    cohorts = {}
    for name, keys in [('all_cases', all_keys), ('valid_initialization', valid_keys)]:
        cohorts[name] = dict(case_keys=[list(k) for k in sorted(keys)], expected_cases_per_arm=len(keys),
            arms={f'{support}_{objective}': _group([r for r in factorial if r['support'] == support and r['objective'] == objective], keys)
                  for support in SUPPORTS for objective in OBJECTIVES},
            paired_contrasts=_contrasts(factorial, keys))
    hardware = []
    for name in ('hardware.json', 'hardware_metadata.json', 'runtime_metadata.json'):
        metadata, error = _load(out / name)
        if metadata is not None or error:
            hardware.append(dict(path=str(out / name), metadata=metadata, error=error))
    for r in recovery + factorial:
        for key in ('hardware', 'gpu_name', 'device_name', 'runtime_metadata'):
            if key in r['result']:
                hardware.append(dict(path=r['path'], field=key, metadata=r['result'][key]))
    initialization_times = []
    for case in cases:
        preserve = (case.get('initialization') or {}).get('preservation', {})
        elapsed = preserve.get('elapsed_seconds')
        if _finite(elapsed):
            initialization_times.append(dict(index=case['index'], seed=case['seed'], elapsed_seconds=float(elapsed)))
    complete = not missing and not failures and not schema_errors
    summary = dict(format='editor_support_report.v1', time=now(), complete=complete,
        expected=dict(recovery_arms=8, recovery_cases=4, factorial_arms=64, factorial_cases=16, objects=8, seeds=list(SEEDS)),
        found=dict(recovery_results=sum(r['has_result'] for r in recovery), factorial_results=sum(r['has_result'] for r in factorial)),
        missing=missing, failures=failures, schema_errors=schema_errors,
        recovery=dict(groups=recovery_groups, records=recovery,
            development_gate=recovery_groups['mesh']['development_gate_passed'], gate_is_generalization_evidence=False),
        factorial=dict(cohorts=cohorts, records=factorial, cases=cases,
            initialization_status_counts=dict(Counter(r['initialization_status'] for r in cases)),
            structure=_structure(factorial)),
        runtime=dict(hardware_metadata=hardware, hardware_recorded=bool(hardware),
            recovery=_timing(recovery), factorial=_timing(factorial),
            shared_initialization_records=initialization_times,
            shared_initialization_seconds=sum(r['elapsed_seconds'] for r in initialization_times),
            timer_definition='optimization elapsed includes snapshots/artifact saves and excludes GT evaluation; initialization counted once per case'),
        bootstrap=dict(unit='object', resamples=BOOTSTRAPS, seed=PARAMS['random_seed']),
        limitations=LIMITATIONS,
        provenance=dict(out=str(out), scan_gt_used_by_report=False, meshes_loaded_by_report=False,
            models_loaded_by_report=False, reads='small JSON only'),
    )
    summary['conclusions'] = conclusions(summary)
    return summary


def _number(value, digits=6, signed=False):
    if value is None or not _finite(value):
        return '—'
    return format(value, ('+' if signed else '') + f'.{digits}f')


def _interval(value):
    return '—' if not value else '[' + ', '.join(_number(v, signed=True) for v in value) + ']'


def conclusions(summary):
    found = summary['found']
    lines = [('全部预期结果及必需指标已齐备。' if summary['complete'] else
        f'当前为partial：recovery {found["recovery_results"]}/8臂、factorial {found["factorial_results"]}/64臂；'
        f'缺失{len(summary["missing"])}条、运行失败{len(summary["failures"])}条、schema缺项{len(summary["schema_errors"])}条，不能称为完成。')]
    gate = summary['recovery']['development_gate']
    mesh = summary['recovery']['groups']['mesh']
    lines.append(('full-mesh开发门尚待全部4例完成。' if gate is None else
        f'full-mesh开发门{"通过" if gate else "未达到"}：{mesh["passed_cases"]}/4例深度误差下降至少50%。') +
        '该诊断不检验独立物体泛化。')
    stats = summary['factorial']['cohorts']['all_cases']['arms']['expanded_mesh']['improvements']['original']['accepted']['chamfer_l1']
    if stats['case_count']:
        lines.append(f'expanded+mesh接受后相对原mesh的CD平均改善为{_number(stats["mean"], signed=True)}，'
            f'物体bootstrap 95%区间{_interval(stats["object_bootstrap_ci95"])}；'
            f'依据{stats["case_count"]}/16案例、{stats["object_count"]}/8物体。正值为改善。')
    else:
        lines.append('尚无完整factorial评价数值，暂不能判断两种loss或SS扩展的效果。')
    init_failures = summary['factorial']['initialization_status_counts'].get('init_failure', 0)
    if init_failures:
        lines.append(f'{init_failures}/16案例保形初始化失败；它们保留在总体并从valid-init子集排除，不能据此宣称SS扩展无效。')
    lines.append('本轮没有新增独立数据；holdout只排除本轮edit，原生成与区域选择已见全部4视图。')
    return lines


def render(summary):
    lines = ['# 第五轮：可微mesh损失与局部SS扩展诊断', '', f'时间：{summary["time"]}', '',
        *[line + '\n' for line in summary['conclusions']],
        '## Known-latent recovery开发诊断', '',
        '|损失|完成/预期|下降≥50%|平均深度误差下降|开发门|',
        '|---|---:|---:|---:|---|']
    for objective, row in summary['recovery']['groups'].items():
        gate = row['development_gate_passed']
        lines.append(f'|{objective}|{row["complete_cases"]}/4|{row["passed_cases"]}/4|'
            f'{_number(row["mean_depth_reduction"], 4)}|{"待完成" if gate is None else ("通过" if gate else "未达到")}|')
    lines += ['', 'mesh至少3/4达到50%下降是开发门。四例目标由decoder生成，不能作为泛化成功率。', '',
        '## 2×2：双基线评价', '',
        '正值均为改善；CD=baseline−endpoint，F@.02=endpoint−baseline。各表均保留无输入约束的无操作案例。', '']
    for cohort, label in [('all_cases', '全部案例'), ('valid_initialization', 'valid-init子集')]:
        rows = summary['factorial']['cohorts'][cohort]
        lines += [f'### {label}', '',
            f'该子集每臂预期{rows["expected_cases_per_arm"]}个object×seed案例。valid-init含有效保形、无新增支持与无输入约束的无操作案例。', '',
            '|臂|结果/预期|接受修改|no_input|proposal相对原CD / F|accepted相对原CD / F|accepted相对step0 CD / F|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in rows['arms'].items():
            def pair(baseline, endpoint):
                values = row['improvements'][baseline][endpoint]
                return _number(values['chamfer_l1']['mean'], signed=True) + ' / ' + _number(values['fscore_0p02']['mean'], 4, True)
            lines.append(f'|{name}|{row["found_results"]}/{row["expected_cases"]}|{row["accepted_edits"]}|'
                f'{row["no_input_cases"]}|{pair("original", "proposal")}|{pair("original", "accepted")}|{pair("step0", "accepted")}|')
        lines += ['', '|臂|accepted相对原CD 95%区间|step0相对原CD / F（初始化漂移）|',
            '|---|---:|---:|']
        for name, row in rows['arms'].items():
            original = row['improvements']['original']
            lines.append(f'|{name}|{_interval(original["accepted"]["chamfer_l1"]["object_bootstrap_ci95"])}|'
                f'{_number(original["step0"]["chamfer_l1"]["mean"], signed=True)} / '
                f'{_number(original["step0"]["fscore_0p02"]["mean"], 4, True)}|')
        lines += ['', '|配对对比（accepted，相对原baseline）|CD改善差|95%物体区间|F改善差|', '|---|---:|---:|---:|']
        for name, contrast in rows['paired_contrasts'].items():
            values = contrast['original']['accepted']
            lines.append(f'|{name}|{_number(values["chamfer_l1"]["mean"], signed=True)}|'
                f'{_interval(values["chamfer_l1"]["object_bootstrap_ci95"])}|{_number(values["fscore_0p02"]["mean"], 4, True)}|')
        lines.append('')
    statuses = summary['factorial']['initialization_status_counts']
    lines += ['## 初始化、结构与计时', '', '初始化case状态：' + '，'.join(f'{k}={v}' for k, v in statuses.items()) + '。',
        'raw/preserved完整漂移、支持扩展receipt、逐例结构与所有状态见summary.json；结构比较基于原mesh、proposal和已记录accepted。', '',
        '|阶段|有计时记录|优化秒数和|平均秒数|最大peak allocated GiB|', '|---|---:|---:|---:|---:|']
    for phase in ('recovery', 'factorial'):
        row = summary['runtime'][phase]
        peak = row['max_peak_allocated_bytes']
        lines.append(f'|{phase}|{row["recorded_cases"]}|{_number(row["sum_optimization_seconds"], 2)}|'
            f'{_number(row["mean_optimization_seconds"], 2)}|{_number(peak / 2**30 if peak is not None else None, 2)}|')
    lines += ['', f'共享保形初始化耗时单独计一次/case：{_number(summary["runtime"]["shared_initialization_seconds"], 2)}秒。',
        'optimization计时包含snapshot和保存，不含GT评价；时间和不是并行运行的总墙钟。',
        '硬件：' + ('已保存JSON metadata，详见summary.json。' if summary['runtime']['hardware_recorded'] else '现有JSON未记录GPU型号，不作推断。'), '',
        '## 缺失与失败', '',
        f'缺失{len(summary["missing"])}条，运行失败{len(summary["failures"])}条，schema缺项{len(summary["schema_errors"])}条。']
    for label, entries in [('缺失', summary['missing']), ('失败', summary['failures']), ('schema', summary['schema_errors'])]:
        for row in entries:
            case = '/'.join(str(row.get(k)) for k in ('phase', 'index', 'seed', 'support', 'objective') if k in row)
            detail = row.get('error') or row.get('fields') or row.get('input_selected_status') or ''
            lines.append(f'- {label}：{case} {detail}')
    lines += ['', '## 适用范围与限制', '', *['- ' + line for line in summary['limitations']], '',
        '[完整JSON、逐例状态与物体bootstrap](summary.json)。', '']
    return '\n'.join(lines)


def generate(out=OUT, store=STORE, final=False, no_audit=False):
    """Write data-root artifacts; append audit only for an explicitly final complete run."""
    out, store = Path(out), Path(store)
    summary = summarize(out)
    summary['report_mode'] = 'final' if final else 'partial'
    summary['audit_appended'] = False
    text = render(summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPORT.md').write_text(text)
    (store / '第五次验证结论.md').write_text(text.replace('(summary.json)', f'({out / "summary.json"})'))
    if final and summary['complete'] and not no_audit:
        payload = dict(found=summary['found'], recovery=summary['recovery']['groups'],
            factorial=summary['factorial']['cohorts'], statuses=summary['factorial']['initialization_status_counts'])
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        marker = f'<!-- editor_support_diagnosis_20261026:{fingerprint} -->'
        audit = store / '审计.md'
        existing = audit.read_text() if audit.exists() else ''
        if marker not in existing:
            addition = f'\n\n# 5，第五轮\n\n时间：{summary["time"]}\n\n{marker}\n\n' + \
                '\n\n'.join(summary['conclusions']) + f'\n\n完整报告：{out / "REPORT.md"}\n完整数值：{out / "summary.json"}\n'
            with audit.open('a') as stream:
                stream.write(addition)
            summary['audit_appended'] = True
        else:
            summary['audit_already_present'] = True
    write(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--partial', action='store_true', help='default: partial report, never append audit')
    mode.add_argument('--final', action='store_true', help='append audit only after complete 8+64 arms; fail otherwise')
    parser.add_argument('--no-audit', action='store_true')
    args = parser.parse_args()
    summary = generate(final=args.final, no_audit=args.no_audit)
    print(json.dumps(dict(complete=summary['complete'], found=summary['found'],
        missing=len(summary['missing']), failures=len(summary['failures']), schema_errors=len(summary['schema_errors']),
        audit_appended=summary['audit_appended'], report=str(OUT / 'REPORT.md')), ensure_ascii=False), flush=True)
    if args.final and not summary['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
