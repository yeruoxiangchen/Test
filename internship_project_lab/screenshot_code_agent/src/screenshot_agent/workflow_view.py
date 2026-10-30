"""Portable local delivery pages. All model/repository content is escaped text."""
from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

LABELS = {'starting':'准备中','snapshot':'复制工作区','reading':'读取截图','indexing':'建立代码索引',
          'planning':'生成执行计划','locate':'定位相关代码','reproduce':'复现与检查','repair':'修改代码',
          'retry':'根据测试反馈修正','public_check':'检查公共测试','review':'分析失败原因',
          'grading':'独立验证补丁','completed':'已结束','paused':'已暂停','error':'运行异常',
          'verified':'验证通过','unresolved':'验证未通过','running':'运行中','pending':'待执行',
          'observed':'已执行，证据已记录','passed':'测试通过','failed':'测试未通过',
          'not_started':'尚未启动','incomplete':'缺少完成证据','returned':'阶段已返回，待验证',
          'inconclusive':'验证条件不足','missing':'尚无复现证据','unverified':'尚未验证'}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else (default or {})


def text(value):
    if not isinstance(value,str):
        value = json.dumps(value,ensure_ascii=False,indent=2)
    return html.escape(value,quote=True)


def page(title, body, *, live=False):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + (
        '<meta http-equiv="refresh" content="15">' if live else '') + f'<title>{text(title)}</title>' + '''
<style>
:root{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;color:#1c2c43;background:#edf2f7}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:32px 24px 60px}
header{border-bottom:3px solid #157f83;margin-bottom:24px;padding-bottom:24px}h1{font-size:30px;margin:8px 0}
h2{font-size:19px;margin:0 0 16px}p{line-height:1.7}small,.muted{color:#62758b}.eyebrow{letter-spacing:.13em;color:#157f83;font-weight:700;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:22px 0}
.card,section{background:white;border:1px solid #dce4ec;border-radius:12px;padding:21px}section{margin:18px 0}
.value{font-size:24px;font-weight:700;margin-top:8px}.badge{display:inline-block;background:#e7f5f0;color:#137553;padding:5px 12px;border-radius:20px;font-size:13px}
.warn{background:#fff1dc;color:#805411}a{color:#09696f}.button{display:inline-block;padding:10px 17px;border-radius:7px;background:#146f78;color:white;text-decoration:none;margin:5px 8px 5px 0}
pre{background:#f4f7fa;border:1px solid #e0e7ef;border-radius:7px;padding:15px;overflow:auto;font:13px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:560px}
summary{cursor:pointer;font-weight:600;padding:12px 0}table{width:100%;border-collapse:collapse;text-align:left;font-size:14px}td,th{padding:12px 8px;border-bottom:1px solid #e4eaf0;vertical-align:top}
ol{padding-left:24px}li{padding:8px 0}.screens{display:flex;flex-wrap:wrap;gap:18px}.screens figure{margin:0;max-width:460px}.screens img{max-width:100%;border-radius:8px;border:1px solid #dce4ec}
input{padding:10px;border:1px solid #c7d5e1;border-radius:6px;width:min(420px,100%)}.diff-add{color:#126d42}.diff-del{color:#a12838}
@media(max-width:600px){main{padding:20px 12px}h1{font-size:24px}.card,section{padding:15px}}
</style><main>''' + body + '</main></html>'


def render_run(directory):
    directory = Path(directory)
    state = read_json(directory/'workflow.json')
    result = read_json(directory/'result.json')
    budget = read_json(directory/'budget.json').get('usage',{})
    progress = read_json(directory/'progress.json')
    status = result.get('status',state.get('status','starting'))
    stage = state.get('stage','starting')
    demo = state.get('model_inference') is False
    title = '截图与报错 · 代码修复工作台'
    tag = '脚本回放演示 · 不代表模型修复成绩' if demo else '仓库级代码修复'
    body = f'<header><div class="eyebrow">SCREENSHOT CODE AGENT</div><h1>{title}</h1><p>{text(tag)}</p>'
    body += f'<span class="badge {"warn" if status in {"error","unresolved","paused"} else ""}">{text(LABELS.get(status,status))}</span>'
    body += f' <span class="muted">{text(state.get("task_id",""))}</span></header>'
    metrics = [('当前阶段',LABELS.get(stage,stage)),('模型调用',budget.get('code_model_calls',0)),
               ('工具调用',budget.get('tool_calls',0)),('累计运行',f"{budget.get('elapsed_seconds',0)/60:.1f} 分钟")]
    body += '<div class="cards">'+''.join(f'<div class="card"><small>{text(k)}</small><div class="value">{text(str(v))}</div></div>' for k,v in metrics)+'</div>'
    body += '<section><h2>任务</h2><pre>'+text(state.get('problem',''))+'</pre>'
    if progress:
        body += '<p class="muted">最近活动：'+text(progress.get('description',''))+'</p>'
    body += '</section><section><h2>执行计划与记录</h2><ol>'
    for row in state.get('plan',[]):
        body += '<li><strong>'+text(LABELS.get(row['phase'],row['phase']))+'</strong> · '+text(row.get('description',''))
        body += '<br><small>'+text(LABELS.get(row.get('status','pending'),row.get('status','pending')))+'</small></li>'
    body += '</ol><p class="muted">执行过一个步骤不等于修复成功；最终状态来自独立副本中的验证。</p></section>'
    evidence = state.get('readings',[])
    images = sorted((directory/'screenshots').glob('*.png')) if (directory/'screenshots').exists() else []
    if images or evidence:
        body += '<section><h2>截图与识别结果</h2><div class="screens">'
        for image in images:
            body += f'<figure><a href="screenshots/{text(image.name)}"><img src="screenshots/{text(image.name)}" alt="输入截图"></a><figcaption>{text(image.stem)}</figcaption></figure>'
        body += '</div><details><summary>查看识别文本与不确定项</summary><pre>'+text(evidence)+'</pre></details></section>'
    if (directory/'repository_map.json').exists():
        body += '<section><details><summary>代码定位与仓库结构</summary><pre>'+text(read_json(directory/'repository_map.json').get('output',''))+'</pre></details></section>'
    body += '<section><h2>修改与交付</h2>'
    patch = directory/'patch.diff'
    if patch.exists():
        body += '<a class="button" href="patch.diff" download>下载补丁</a>'
        if (directory/'delivery.zip').exists():
            body += '<a class="button" href="delivery.zip" download>下载交付包</a>'
        lines = patch.read_text(errors='replace').splitlines()
        body += '<pre>'+ '\n'.join('<span class="'+('diff-add' if line.startswith('+') else 'diff-del' if line.startswith('-') else '')+'">'+text(line)+'</span>' for line in lines[:2500])+'</pre>'
        if len(lines)>2500:
            body += '<p>预览已截断，完整修改见下载的补丁。</p>'
    else:
        body += '<p class="muted">执行过程中保留工作区，完成修复后导出补丁。</p>'
    body += '</section><section><h2>测试与反馈</h2>'
    for item in state.get('checks',[]):
        body += '<details><summary>'+text(item.get('label','公共测试'))+' · '+('通过' if item.get('passed') else '未通过')+'</summary>'
        if 'regression' in item:
            body += '<p>原有功能回归：'+text(LABELS.get(item['regression']['status'],item['regression']['status']))
            body += '；原问题复现验证：'+text(LABELS.get(item['reproduction']['status'],item['reproduction']['status']))+'</p>'
        known = item.get('known_issue', {})
        if known and known.get('status') != 'not_configured':
            body += '<p>已知问题开发验收：'+text(LABELS.get(known['status'], known['status']))+'</p>'
            body += '<pre>'+text(known.get('output', ''))+'</pre>'
        body += '<pre>'+text(item.get('output',''))+'</pre></details>'
    for i, review in enumerate(state.get('reviews',[]),1):
        body += f'<details><summary>第{i}次失败分析</summary><pre>'+text(review)+'</pre></details>'
    if result:
        body += '<p>最终结果：<strong>'+text(LABELS.get(result.get('status',''),result.get('status','')))+'</strong></p>'
        body += '<p class="muted">'+text(result.get('verification_scope',''))+'</p>'
        if result.get('error'):
            body += '<pre>'+text(result['error'])+'</pre>'
        if result.get('agent_error'):
            body += '<pre>执行异常：'+text(result['agent_error'])+'</pre>'
        if result.get('validation'):
            body += '<details><summary>最终验证详情</summary><pre>'+text(result['validation'])+'</pre></details>'
    body += '</section><p class="muted">报告保存在本地。运行日志与调用记录可用于检查失败原因。</p>'
    out = directory/'report.html'
    temporary = directory/'report.html.tmp'
    temporary.write_text(page(title,body,live=status=='running'),encoding='utf-8')
    temporary.replace(out)
    return out


def export_delivery(directory):
    directory = Path(directory)
    # Only user-facing artifacts. Never bundle hidden reference metadata,
    # baseline repositories, caches, credentials or complete benchmark records.
    paths = [directory/name for name in ('report.html','patch.diff','result.json') if (directory/name).is_file()]
    paths += sorted((directory/'screenshots').glob('*.png'))
    with zipfile.ZipFile(directory/'delivery.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.name == 'report.html':
                # The downloaded package must not contain a broken self-download link.
                body = path.read_text().replace('<a class="button" href="delivery.zip" download>下载交付包</a>','')
                archive.writestr('report.html',body)
            else:
                archive.write(path,str(path.relative_to(directory)))
    return directory/'delivery.zip'


def render_batch(directory, rows, total):
    directory = Path(directory)
    finished = sum(r.get('status') in {'verified','unresolved','error'} for r in rows)
    success = sum(bool(r.get('resolved')) for r in rows)
    body = '<header><div class="eyebrow">SCREENSHOT CODE AGENT</div><h1>修复任务进度与结果</h1>'
    body += f'<p>已结束 {finished}/{total} · 独立验证通过 {success}/{total}</p></header>'
    config = read_json(directory/'workflow_config.json')
    if config.get('workflow', {}).get('development_casebook'):
        body += '<p>已知失败题开发回归：使用先前失败反馈和可见验收检查，全部任务保留。结果不代表未见任务的泛化成功率。</p>'
    body += '<input id="query" placeholder="按任务名称或状态筛选" aria-label="筛选任务"><section><table><thead><tr><th>任务</th><th>状态</th><th>模型调用</th><th>报告</th></tr></thead><tbody id="tasks">'
    for row in rows:
        task_id = row['task_id']
        body += '<tr><td>'+text(task_id)+'</td><td>'+text(LABELS.get(row.get('status','pending'),row.get('status','pending')))+'</td>'
        body += '<td>'+text(str(row.get('usage',{}).get('code_model_calls','—')))+'</td>'
        body += f'<td><a href="runs/{text(task_id)}/report.html">查看</a></td></tr>'
    body += '</tbody></table></section><script>document.getElementById("query").addEventListener("input",function(){for(const row of document.querySelectorAll("#tasks tr")){row.hidden=!row.textContent.toLowerCase().includes(this.value.toLowerCase());}});</script>'
    (directory/'report.html').write_text(page('修复任务结果',body,live=finished<total),encoding='utf-8')
