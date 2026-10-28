"""Reports only observed measurements. Empty quality labels remain empty."""
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import random
import statistics

from .support import SOURCE, read, save


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summary(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'p95': percentile(values, .95), 'min': min(values, default=None), 'max': max(values, default=None)}


def timings(record):
    def diff(a, b):
        if record.get(a) is None or record.get(b) is None:
            return None
        return record[a] - record[b]
    return {'queue_s': diff('claimed', 'accepted'),
            'dispatch_s': diff('execution_start', 'claimed'),
            'execution_s': diff('execution_end', 'execution_start'),
            'save_poll_s': diff('output_ready', 'execution_end'),
            'end_to_end_s': diff('output_ready', 'accepted'),
            'client_received_s': diff('client_received', 'attempt_started')}


def eligible(record):
    return bool(record.get('success') and record.get('resources', {}).get('clean_device_observed')
                and not record.get('resumed') and 'sample' not in record.get('cached_nodes', [])
                and record.get('execution_start') is not None and record.get('execution_end') is not None)


def aggregate(records):
    groups = {}
    last_model = {}
    contexts = {}
    for record in sorted(records, key=lambda r: r.get('execution_start') or float('inf')):
        worker = record.get('worker_id')
        model = 'sdxl' if record.get('params', {}).get('mode') == 'text-to-image' else 'wan22'
        context = ('first_observed' if worker not in last_model else
                   'model_switch' if last_model[worker] != model else 'same_model_sequence')
        if record.get('phase') in {'one_gpu', 'two_gpu'}:
            context = 'after_explicit_warmup'
        contexts[record['id']] = context
        if record.get('execution_start') is not None:
            last_model[worker] = model
    for record in records:
        p = record.get('params', {})
        key = '|'.join(str(x) for x in (record.get('phase'), record.get('route', 'studio'), p.get('mode'),
                                       p.get('width'), p.get('height'), p.get('steps'),
                                       None if p.get('mode') == 'text-to-image' else p.get('frames'), contexts[record['id']]))
        group = groups.setdefault(key, {'completed': 0, 'successful': 0, 'dependency_failures': 0, 'performance_rows': []})
        group['completed'] += bool(record.get('finished'))
        group['successful'] += bool(record.get('success'))
        group['dependency_failures'] += bool(record.get('dependency_failed'))
        if eligible(record):
            group['performance_rows'].append({**timings(record),
                'observed_process_peak_mib': record['resources'].get('observed_process_peak_mib')})
    for group in groups.values():
        rows = group.pop('performance_rows')
        group['valid_performance_samples'] = len(rows)
        group['metrics'] = {key: summary([r[key] for r in rows]) for key in rows[0]} if rows else {}
    paired = {}
    for record in records:
        if record.get('phase') == 'paired' and eligible(record):
            paired.setdefault(record['id'].rsplit('-', 1)[0], {})[record['route']] = record
    deltas = []
    for case, pair in paired.items():
        if set(pair) == {'direct', 'studio'}:
            direct, studio = timings(pair['direct']), timings(pair['studio'])
            deltas.append({'case': case, 'client_received_delta_s': studio['client_received_s'] - direct['client_received_s'],
                           'execution_delta_s': studio['execution_s'] - direct['execution_s']})
    return {'groups': groups, 'paired_deltas': deltas,
            'paired_delta_summary_s': summary([p['client_received_delta_s'] for p in deltas]),
            'context_note': 'Same-model sequence is observational; it does not prove all weights remained resident.'}


def gallery(root, records):
    rng = random.Random(hashlib.sha256(root.name.encode()).hexdigest())
    dataset = [r for r in records if r.get('phase') == 'dataset' and r.get('success')]
    rng.shuffle(dataset)
    mapping, cards = {}, []

    def media(record, reference=False):
        path = record.get('reference') if reference else record.get('output')
        if not path:
            return ''
        try:
            url = str(Path(path).relative_to(root))
        except ValueError:
            return ''
        url = html.escape(url, quote=True)
        if str(path).endswith('.mp4'):
            return f'<video src="{url}" controls preload="metadata"></video>'
        return f'<img src="{url}" loading="lazy" alt="{"参考图" if reference else "生成结果"}">'

    def select(name, labels):
        return '<select data-field="' + html.escape(name) + '"><option value="">未评分</option>' + ''.join(
            f'<option value="{value}">{label}</option>' for value, label in labels) + '</select>'

    for index, record in enumerate(dataset, 1):
        ident = f'item-{index:03d}'
        mapping[ident] = record['id']
        labels = [('yes', '满足'), ('no', '不满足'), ('unclear', '无法判断')]
        criteria = [(f'check_{i}', text) for i, text in enumerate(record.get('checks', []))]
        criteria += [('structure', '无严重结构错误'), ('usable', '主要要求满足且可用')]
        if record['params']['mode'] != 'text-to-image':
            criteria += [('motion', '要求的运动发生'), ('consistency', '主体与背景保持合理稳定')]
        if record.get('reference'):
            criteria += [('reference', '保留输入主体')]
        controls = ''.join(f'<label>{html.escape(title)} {select(field, labels)}</label>' for field, title in criteria)
        cards.append(f'<article data-id="{ident}"><h2>{ident} · {html.escape(record["params"]["mode"])}</h2>'
                     + (f'<details><summary>参考图</summary>{media(record, True)}</details>' if record.get('reference') else '')
                     + media(record) + f'<p>{html.escape(record["params"]["prompt"])}</p>{controls}<textarea data-field="note" placeholder="判断依据与失败现象"></textarea></article>')
    pairs = {}
    for record in records:
        if record.get('phase') == 'paired' and record.get('success'):
            pairs.setdefault(record['id'].rsplit('-', 1)[0], []).append(record)
    for index, pair in enumerate(pairs.values(), 1):
        if len(pair) != 2:
            continue
        rng.shuffle(pair)
        ident = f'pair-{index:03d}'
        mapping[ident] = {'A': pair[0]['id'], 'B': pair[1]['id']}
        cards.append(f'<article class="pair" data-id="{ident}"><h2>{ident}</h2><div class="side"><div>A{media(pair[0])}</div><div>B{media(pair[1])}</div></div>'
                     + f'<p>{html.escape(pair[0]["params"]["prompt"])}</p>'
                     + select('preference', [('A', '更偏好 A'), ('B', '更偏好 B'), ('tie', '相当'), ('neither', '均不可用'), ('unclear', '无法判断')])
                     + '<textarea data-field="note" placeholder="偏好依据"></textarea></article>')
    save(root / 'review-mapping.json', mapping)
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Studio 评测审查</title>
<style>body{font:16px system-ui;background:#f4f5f7;color:#17232d;margin:24px auto;max-width:1400px;padding:0 20px}header{padding:20px;background:white;border-radius:12px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-top:24px}article{background:white;padding:16px;border-radius:12px}h2{font-size:18px}img,video{width:100%;max-height:360px;object-fit:contain;background:#e9ebee}label{display:block;margin:10px 0}select,input,textarea,button{font:inherit;padding:8px;border:1px solid #bcc4cf;border-radius:6px}textarea{display:block;box-sizing:border-box;width:100%;min-height:80px;margin-top:12px}.pair{grid-column:span 2}.side{display:flex;gap:12px}.side>div{width:50%}@media(max-width:720px){.pair{grid-column:span 1}}</style>
<header><h1>生成作品审查</h1><p>按提示词检查内容，再判断结构、运动及参考一致性。视频必须播放，不能只看缩略图。留空表示未评分。评分暂存在本浏览器，导出后保留文件；此页面不自动上传评分。</p>
<label>评审者 <input id="reviewer" placeholder="填写姓名或标识"></label><label>评审类型 <select id="kind"><option value="human">人工审查</option><option value="assistant">助手视觉审查</option></select></label><button id="export">导出评分 JSON</button><span id="status"></span></header><main>__CARDS__</main>
<script>const run=__RUN__,key='studio-review:'+run;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};
const reviewer=document.querySelector('#reviewer'),kind=document.querySelector('#kind');reviewer.value=saved.reviewer||'';kind.value=saved.kind||'human';
for(const card of document.querySelectorAll('[data-id]'))for(const field of card.querySelectorAll('[data-field]'))field.value=(saved.items?.[card.dataset.id]||{})[field.dataset.field]||'';
function collect(){const items={};for(const card of document.querySelectorAll('[data-id]')){items[card.dataset.id]={};for(const field of card.querySelectorAll('[data-field]'))items[card.dataset.id][field.dataset.field]=field.value}return {run,reviewer:reviewer.value.trim(),kind:kind.value,updated:new Date().toISOString(),items}};
document.addEventListener('change',()=>{try{localStorage.setItem(key,JSON.stringify(collect()));document.querySelector('#status').textContent=' 已保存在当前浏览器'}catch(e){document.querySelector('#status').textContent=' 浏览器存储不可用，请导出'}});
document.querySelector('#export').onclick=()=>{const data=collect();if(!data.reviewer){reviewer.focus();document.querySelector('#status').textContent=' 请填写评审者';return}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=run+'-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script></html>'''
    (root / 'review.html').write_text(template.replace('__CARDS__', ''.join(cards) or '<p>尚无生成成功的作品，当前没有质量评分。</p>').replace('__RUN__', json.dumps(root.name).replace('<', '\\u003c')))


def render(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = [read(path) for path in sorted((root / 'tasks').glob('*.json'))]
    protocol = read(root / 'environment.json', {}).get('protocol') or read(SOURCE / 'evaluation/protocol.json')
    faults = read(root / 'faults/results.json', {}).get('results', [])
    state = read(root / 'state.json', {'phase': 'not_started'})
    metrics = aggregate(records)
    save(root / 'metrics.json', metrics)
    dataset = [r for r in records if r.get('phase') == 'dataset' and r.get('finished')]
    wanted = {'text-to-image': len(protocol['image_cases']) * len(protocol['seeds']),
              'text-to-video': len(protocol['video_cases']) * len(protocol['seeds']), 'image-to-video': 10 * len(protocol['seeds'])}
    lines = ['# Studio 实测报告', '', f'更新：{datetime.now(timezone.utc).isoformat(timespec="seconds")}；状态：`{state["phase"]}`。', '',
             '这是内部工程验收与自定义诊断集，不是官方 GenEval、VBench 或 MLPerf 成绩。未执行项目不填数值，成功率保留失败分母。', '',
             '## 执行覆盖', '', '| 项目 | 计划数 | 已结束 | 解码及规格通过 |', '|---|---:|---:|---:|']
    for mode, expected in wanted.items():
        subset = [r for r in dataset if r['params']['mode'] == mode]
        lines.append(f'| {mode} | {expected} | {len(subset)} | {sum(bool(r.get("success")) for r in subset)} |')
    smoke = [r for r in records if r.get('phase') == 'smoke' and r.get('finished')]
    lines += ['', f'小规格真实生成检查：已结束 {len(smoke)}/3，通过 {sum(bool(r.get("success")) for r in smoke)} 项。', '',
              '## 服务故障验收', '', f'真实 API + CPU 上游模拟器：{sum(bool(r["passed"]) for r in faults)}/{len(faults)} 通过。此项不代表真实 GPU 生成成功率。', '',
              '| 场景 | 结果 |', '|---|---|']
    lines += [f'| {r["case"]} | {"通过" if r["passed"] else "失败"} |' for r in faults]
    lines += ['', '完整证据见 `faults/results.json`。', '', '## 性能', '',
              '仅汇总采样期间未发现其他计算进程、未断点续接、未复用采样结果的成功任务。按模式、规格和执行阶段分组；P95 是小样本经验分位数，不代表服务保障。', '',
              '| 分组（阶段/路径/模式/宽/高/步数/帧数/运行上下文） | 有效 n | 端到端中位数 s | P95 s | 执行中位数 s | 观测进程显存峰值 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.2f}'
    for key, group in metrics['groups'].items():
        m = group['metrics']
        e2e = m.get('end_to_end_s', {})
        lines.append('| ' + key.replace('|', ' / ') + f' | {group["valid_performance_samples"]} | {number(e2e.get("median"))} | {number(e2e.get("p95"))} | {number(m.get("execution_s", {}).get("median"))} | {number(m.get("observed_process_peak_mib", {}).get("max"))} |')
    if not records:
        lines += ['| 尚无真实生成记录 | 0 | — | — | — | — |']
    pair = metrics['paired_delta_summary_s']
    lines += ['', f'直接调用/工作台配对：有效 {pair["n"]}/8 对，客户端完整接收时延差（工作台−直接）中位数 {number(pair["median"])} 秒。差值包括执行波动；`metrics.json` 同时保留执行时长差，不能直接解释为纯框架开销。', '',
              '显存每秒观测一次，可能漏掉瞬时峰值；同模型连续运行不保证所有权重始终驻留，执行事件包含模型加载、文本编码、采样、解码和编码。', '', '## 吞吐与扩展', '']
    throughput = {}
    for label in ['one_gpu', 'two_gpu']:
        result = read(root / (label + '.json'))
        throughput[label] = result
        if result:
            lines.append(f'- {label}：{result["successful"]}/{result["tasks"]} 有效完成，窗口 {result["seconds"]:.2f} s，{result["images_per_hour"]:.2f} 张/小时；资源无其他计算进程：{result["all_devices_clean"]}。')
        else:
            lines.append(f'- {label}：未完成。')
    a, b = throughput.values()
    if a and b and a['all_devices_clean'] and b['all_devices_clean'] and a['images_per_hour'] > 0:
        lines += [f'- 两卡扩展效率：{b["images_per_hour"] / (2*a["images_per_hour"]):.3f}。']
    lines += ['', '吞吐窗口包含下载和完整解码验收，各条件只测 8 张，是探索性固定批次，不是稳定负载容量认证。', '',
              '## 质量审查', '', '已生成的作品与逐条检查表见 [审查页面](review.html)。目前没有导入独立评审结果，不填语义质量分数。像素变化和黑帧检测只用于故障诊断；两个种子共享同一提示词，不能当成完全独立的语义样本。', '',
              '## 方法依据', '',
              '- [GenEval](https://github.com/djghosh13/geneval)：对象、计数、颜色、位置及属性绑定。',
              '- [VBench](https://github.com/Vchitect/VBench) 与 [VBench++](https://arxiv.org/abs/2411.13503)：视频一致性、运动及图生视频条件遵循。',
              '- [MLPerf Inference 规则](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)：明确负载、计时边界和吞吐场景。本项目未运行 LoadGen。', '',
              '协议、样本和代码摘要见 `environment.json`，逐任务原始记录见 `tasks/`，实际执行事件见 `histories/`，每秒资源观测见 `telemetry.jsonl`。']
    if state.get('error'):
        lines += ['', '当前错误：', '```', state['error'], '```']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    output = io.StringIO()
    fields = ['id', 'phase', 'route', 'success', 'performance_eligible', 'error', *timings({}).keys()]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({**{k: record.get(k) for k in ['id', 'phase', 'route', 'success', 'error']},
                         'performance_eligible': eligible(record), **timings(record)})
    (root / 'tasks.csv').write_text(output.getvalue())
    gallery(root, records)
