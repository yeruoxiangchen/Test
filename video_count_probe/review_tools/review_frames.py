"""Read-only video review preparation and separately labelled AI pilot reports.

Never changes generated videos, frozen plans, or the human summary.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = Path('/data/zjr/video_count_probe_20261025/pilot')
OUT = ROOT / 'ai_review_20261027'
FFMPEG = Path('/home/zjr/Tracker/video_count_probe/.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def frames(video):
    process = subprocess.run([
        str(FFMPEG), '-v', 'error', '-threads', '1', '-i', str(video),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'
    ], capture_output=True, check=True)
    stride = 832 * 480 * 3
    if len(process.stdout) != 81 * stride:
        raise ValueError(f'Unexpected decoded frame size: {video}')
    return [Image.frombytes('RGB', (832, 480), process.stdout[i*stride:(i+1)*stride]) for i in range(81)]


def sheet(images, indices, path, title, *, crop=None, width=320, cols=7):
    if crop is not None:
        images = [im.crop(crop) for im in images]
    height = round(width * images[0].height / images[0].width)
    cell_h = height + 22
    rows = (len(indices) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * width, 30 + rows * cell_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='black')
    for j, index in enumerate(indices):
        x, y = (j % cols) * width, 30 + (j // cols) * cell_h
        draw.text((x + 4, y + 3), f'f{index:02d}  {index/16:.4f}s', fill='black')
        canvas.paste(images[index].resize((width, height), Image.Resampling.LANCZOS), (x, y + 22))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def prepare():
    plan = json.loads((ROOT/'plan.json').read_text())
    cases = {c['id']: c for c in plan['cases']}
    jobs = list(plan['jobs'])
    random.Random(82415).shuffle(jobs)
    private, blind = [], []
    for index, job in enumerate(jobs, 1):
        record = json.loads((ROOT/'records'/f"{job['id']}.json").read_text())
        video = ROOT/record['video']
        if hashlib.sha256(video.read_bytes()).hexdigest() != record['video_sha256']:
            raise ValueError(f'Video digest mismatch: {job["id"]}')
        destination = OUT/'sheets'/f'{index:03d}.jpg'
        case = cases[job['case_id']]
        if not destination.exists():
            sheet(frames(video), list(range(0,81,2)), destination,
                  f"Blind clip {index:03d} | action={case['action']} | sampled 8 fps, original 16 fps")
        private.append({'blind_id':index,'job_id':job['id'],'video':record['video'],
                        'video_sha256':record['video_sha256']})
        blind.append({'blind_id':index,'action':case['action'],'criterion':case['criterion'],
                      'sheet':str(destination)})
        if index % 16 == 0:
            print(f'Prepared {index}/{len(jobs)}',flush=True)
    write_json(OUT/'private_mapping.json', {'plan_hash':plan['plan_hash'],'items':private})
    write_json(OUT/'blind_manifest.json', {'plan_hash':plan['plan_hash'],'items':blind})
    write_json(OUT/'protocol.json', {
        'review_type':'AI visual review of time-labelled video frames; not human ground truth',
        'rater':'Codex visual review', 'initial_sampling_fps':8,'original_fps':16,
        'methods_and_targets_hidden_during_annotation':True,
        'blinding_limit':'The reviewer previously saw the experiment design and metadata for two example job IDs; sheets use unrelated serial IDs.',
        'source_generation_unchanged':True,
        'counting':'Complete action cycles only; interrupted cycles excluded; ambiguity explicitly retained.',
        'timing':'Completion timestamps estimated from reviewed frames; sparse-frame estimates are not exact frame labels.',
        'pause':'Judge a clear interval of stopped action between second and third completed cycles, independent of requested condition.',
        'original_human_summary_unchanged':True,
    })
    print(f'Prepared {len(jobs)} blind sheets under {OUT}')


def detail(ids, start, end, crop):
    mapping = {r['blind_id']:r for r in json.loads((OUT/'private_mapping.json').read_text())['items']}
    for index in ids:
        path=OUT/'detail'/f'{index:03d}_{start:02d}_{end:02d}.jpg'
        sheet(frames(ROOT/mapping[index]['video']),list(range(start,end+1)),path,
              f'Blind clip {index:03d}, full frame rate detail',crop=crop,width=320,cols=7)
        print(path)


def annotate(items):
    # Accept JSON from stdin or a shell-quoted command argument.
    mapping={r['blind_id']:r for r in json.loads((OUT/'private_mapping.json').read_text())['items']}
    for item in items:
        index=item['blind_id']
        if index not in mapping:
            raise ValueError('Unknown blind ID')
        for key in ['visually_usable','action_present','ambiguous','pause_respected']:
            if not isinstance(item[key],bool):raise ValueError(key)
        times=item['completion_times']
        if times!=sorted(set(times)) or any(not 0<=v<=5 for v in times):raise ValueError('times')
        if times and not item['action_present']:raise ValueError('action_present')
        if item.get('count_range') and not item['ambiguous']:raise ValueError('range requires ambiguity')
        item.update(complete=True,rater='Codex AI frame review',annotation_source='ai_visual_frame_review')
        write_json(OUT/'labels'/f'{index:03d}.json',item)
    print('AI-labelled:',len(list((OUT/'labels').glob('*.json'))),'/96')


def summarize():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from video_count_probe.evaluation import score_annotation,validate_annotation,completed_records
    from video_count_probe.core import checked_plan,file_sha,source_hash
    human_summary_sha=file_sha(ROOT/'summary.json')
    plan=checked_plan(ROOT/'plan.json')
    records,errors=completed_records(ROOT,plan)
    identity=json.loads((ROOT/'identity.json').read_text())
    if source_hash()!=identity['source_hash']:
        raise ValueError('Generation source has changed')
    cases={c['id']:c for c in plan['cases']}
    jobs={j['id']:j for j in plan['jobs']}
    mapping_data=json.loads((OUT/'private_mapping.json').read_text())
    if mapping_data['plan_hash']!=plan['plan_hash']:
        raise ValueError('Blind mapping/plan mismatch')
    mapping=mapping_data['items']
    annotations=[];scored=[]
    for row in mapping:
        path=OUT/'labels'/f"{row['blind_id']:03d}.json"
        if not path.exists():continue
        label=json.loads(path.read_text())
        job=jobs[row['job_id']];case=cases[job['case_id']]
        record=records[job['id']]
        annotation={**label,'job_id':job['id'],'video_sha256':row['video_sha256']}
        validate_annotation(annotation,record,case,plan)
        annotations.append(annotation)
        scores=score_annotation(annotation,case,plan)
        bounds=label.get('count_range')
        possible=scores['exact_usable']
        if label['ambiguous'] and label['visually_usable']:
            possible=int(bounds is None or bounds[0]<=case['target_count']<=bounds[1])
        scored.append({**job,'blind_id':row['blind_id'],'action':case['action'],
                       'target_count':case['target_count'],'schedule':case['schedule'],
                       **scores,'observed_count':len(label['completion_times']) if scores['judgeable'] else None,
                       'candidate_timestamp_count':len(label['completion_times']),
                       'possible_exact_usable':possible,'video':row['video'],
                       'completion_times':label['completion_times'],'notes':label.get('notes',''),
                       'count_range':bounds})
    methods={};groups={}
    for method in ['count_prompt','timed_prompt','relay_unrouted','relay']:
        part=[r for r in scored if r['method']==method]
        def aggregate(rows):
            keys=['exact_usable','visually_usable','action_present','ambiguous','timing_success']
            result={'n':len(rows),**{k:sum(r[k] for r in rows) for k in keys}}
            result['judgeable']=sum(r['judgeable'] for r in rows)
            result['possible_exact_usable']=sum(r['possible_exact_usable'] for r in rows)
            result['no_complete_cycle_judgeable']=sum(r['observed_count']==0 for r in rows)
            judged=[r['absolute_count_error'] for r in rows if r['absolute_count_error'] is not None]
            result['count_mae_judgeable']=sum(judged)/len(judged) if judged else None
            pause=[r['pause_success'] for r in rows if r['pause_success'] is not None]
            result['pause_success']=sum(pause);result['pause_n']=len(pause)
            return result
        methods[method]=aggregate(part)
        groups[method]={}
        for field in ['action','target_count','schedule']:
            groups[method][field]={str(v):aggregate([r for r in part if r[field]==v]) for v in sorted({r[field] for r in part})}
    paired={}
    for control in ['count_prompt','timed_prompt','relay_unrouted']:
        a={(r['case_id'],r['seed']):r for r in scored if r['method']=='relay'}
        b={(r['case_id'],r['seed']):r for r in scored if r['method']==control}
        pairs=sorted(a.keys()&b.keys())
        for key in pairs:
            ar,br=records[a[key]['id']],records[b[key]['id']]
            if ar['initial_latent_sha256']!=br['initial_latent_sha256']:
                raise ValueError('Paired initial noise mismatch')
            if control=='relay_unrouted' and ar['prompt']!=br['prompt']:
                raise ValueError('Routed/unrouted text mismatch')
        diffs=[a[k]['exact_usable']-b[k]['exact_usable'] for k in pairs]
        paired['relay_vs_'+control]={'pairs':len(pairs),'wins':diffs.count(1),'losses':diffs.count(-1),'ties':diffs.count(0),
                                    'delta':sum(diffs)/len(diffs) if diffs else None}
    # Secondary diagnostic follows notes written while blinded, without changing
    # the frozen complete-cycle metric or the original labels.
    extra_cycle_ids=[r['blind_id'] for r in scored if r['exact_usable'] and r['blind_id'] in (22,45,61)]
    action_totals={a:aggregate([r for r in scored if r['action']==a]) for a in sorted({r['action'] for r in scored})}
    result={'review_type':'AI frame review, not validated human annotations','plan_hash':plan['plan_hash'],
            'expected':96,'ai_reviewed':len(scored),'human_validated':False,'human_metrics_ready':False,
            'methods':methods,'subgroups':groups,'actions':action_totals,'paired':paired,'per_video':scored,
            'integrity':{'verified_videos':len(records),'execution_errors':len(errors),
                         'paired_noise_verified':True,'routed_unrouted_prompts_equal':True,
                         'generation_source_hash':source_hash(),'human_summary_sha256':human_summary_sha,
                         'routed_calls_by_method':{m:sorted({r['routed_calls'] for r in records.values() if r['job']['method']==m}) for m in methods}},
            'annotation_sha256':{p.name:file_sha(p) for p in sorted((OUT/'labels').glob('*.json'))},
            'secondary_diagnostic':{'exact_count_clips_with_extra_incomplete_cycle':extra_cycle_ids,
                                    'interpretation':'These clips pass the frozen completed-cycle count, but start an additional unfinished squat; count match alone does not establish stopping control.'},
            'sensitivity_definition':'possible_exact_usable treats visually usable ambiguous clips as correct only when their blind count range includes the target (or no range is available). Unusable clips remain failures. This is an annotation sensitivity range, not a statistical confidence interval.',
            'limitations':['Single AI reviewer; no independent human validation.',
                           '8 fps overview and selected 16 fps detail; brief events can remain ambiguous.',
                           'Event timestamps are estimates; timing scores are especially provisional.',
                           'One small video model and 24 conditions/seeds per method; descriptive pilot only.']}
    write_json(OUT/'annotations_ai.json',{'plan_hash':plan['plan_hash'],'annotation_source':'ai_visual_frame_review','annotations':annotations})
    write_json(OUT/'summary_ai.json',result)
    write_review_report(result)
    if file_sha(ROOT/'summary.json')!=human_summary_sha:
        raise RuntimeError('Original human summary was modified')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_video','subgroups','annotation_sha256']},ensure_ascii=False,indent=2))
    print(f"AI report: {OUT/'report_ai.html'}")


def write_review_report(result):
    fields=['blind_id','id','method','case_id','seed','action','target_count','observed_count',
            'count_range','exact_usable','possible_exact_usable','visually_usable','ambiguous',
            'completion_times','notes','video']
    with (OUT/'per_video_ai.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in result['per_video']:
            writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],list) else row[k] for k in fields})
    escape=lambda value:html.escape(str(value),quote=True)
    names={'count_prompt':'直接次数提示','timed_prompt':'带时间段的提示',
           'relay_unrouted':'Relay 文本，关闭路由','relay':'Relay 文本，开启路由'}
    actions={'clap':'拍手','jumping_jack':'开合跳','squat':'深蹲'}
    table=[]
    for method,m in result['methods'].items():
        table.append(f"<tr><td>{escape(names[method])}</td><td>{m['exact_usable']}/{m['n']}</td>"
                     f"<td>{m['exact_usable']}–{m['possible_exact_usable']}/{m['n']}</td>"
                     f"<td>{m['visually_usable']}/{m['n']}</td><td>{m['ambiguous']}/{m['n']}</td></tr>")
    cards=[]
    for row in result['per_video']:
        index=row['blind_id'];uncertain=bool(row['ambiguous'] or not row['visually_usable'])
        status='uncertain' if uncertain else ('success' if row['exact_usable'] else 'failure')
        status_name={'uncertain':'不确定 / 不可辨认','success':'完整次数匹配','failure':'完整次数不匹配'}[status]
        count=(f"{row['count_range'][0]}–{row['count_range'][1]}（不确定）" if row['count_range'] else '无法确定') if uncertain else str(row['observed_count'])
        detail_links=' '.join(f'<a href="detail/{escape(p.name)}">原帧率局部图</a>' for p in sorted((OUT/'detail').glob(f'{index:03d}_*.jpg')))
        cards.append(f'''<details class="clip" data-method="{escape(row['method'])}" data-action="{escape(row['action'])}" data-status="{status}">
<summary>#{index:03d} · {actions[row['action']]} · {names[row['method']]} · 目标 {row['target_count']} / 观察 {escape(count)} · {status_name}</summary>
<p>{escape(row['notes'])}</p><p>条件：{escape(row['case_id'])}；种子：{row['seed']}；视频：{escape(row['id'])}</p>
<video controls preload="none" src="../{escape(row['video'])}"></video>
<p>完成时刻（歧义片段仅为候选）：{escape(row['completion_times'])}</p>
<p><a href="sheets/{index:03d}.jpg">8 fps 总览帧图</a> {detail_links} <a href="labels/{index:03d}.json">原始盲评标注</a></p></details>''')
    method_options=''.join(f'<option value="{m}">{n}</option>' for m,n in names.items())
    action_options=''.join(f'<option value="{a}">{n}</option>' for a,n in actions.items())
    paired=result['paired']['relay_vs_relay_unrouted']
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频计数 Pilot — AI 帧评审</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:30px auto;padding:0 20px;color:#192734;background:#fafafa}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ddd;padding:10px;text-align:left}}.notice{{padding:16px;background:#fff3cc;border-left:4px solid #af7c00}}select{{padding:8px;margin:6px}}details{{background:white;border:1px solid #ddd;padding:12px;margin:12px 0}}summary{{cursor:pointer}}video{{width:100%;max-width:832px}}[hidden]{{display:none}}a{{color:#075ea8}}</style>
<h1>视频动作次数实验：96 条 AI 评审</h1>
<p class="notice">已完成 {result['ai_reviewed']}/96 条。由 Codex 检查时间标记帧图，先隐藏方法与目标次数，标注后揭盲；这是 AI 评审，未经过独立人工验证。总览为 8 fps，选定疑难片段为原始 16 fps。原始人工 summary.json 保持不变。</p>
<p>Wan2.1-T2V-1.3B，12 个条件 × 2 个种子 × 4 个方法。主要指标为“可辨认、无歧义、完整动作次数匹配”；所有 24 条均进入各方法分母。</p>
<table><thead><tr><th>方法</th><th>明确计数成功</th><th>考虑计数歧义后的范围</th><th>画面可核验</th><th>存在歧义</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p>上表范围只改变可核验歧义片段的计数判定，不把坏视频算成功；它不是统计置信区间，也不代表真实准确率边界。</p>
<p>关键配对：Relay 开启 / 关闭路由，{paired['pairs']} 对中 {paired['wins']} 胜、{paired['losses']} 负、{paired['ties']} 平。本轮没有显示路由的明确计数收益。</p>
<p>拍手、开合跳均未出现明确的次数匹配。拍手常出现双掌不接触或重复过多；开合跳常变成举手或并脚跳。深蹲有少数完整周期，但半次动作、弯腰与取景问题较多。</p>
<p><strong>指标局限：</strong>3 条明确计数匹配均开始了额外一次未完成的深蹲（盲评编号 22、45、61）。冻结指标只数完整周期，所以匹配并不代表可靠遵守“做完后停止”。此次不改旧指标；后续应单独评价额外未完成动作与停止行为。</p>
<p><strong>对选题的含义：</strong>这轮揭示了当前设置的动作质量与计数控制问题，尚不能证明新方法有效，也不能据此否定视频计数方向。它包含已有方法的移植对照，没有训练后的新方法。下一步应先用简短无次数提示检查基础动作能力，区分提示负担、基座动作能力与时间路由问题，再决定是否扩大实验或训练。</p>
<p>完整性已核验：96 条视频摘要、结果身份、配对初始噪声，以及路由/非路由的相同文本。Relay 每条记录 1500 次路由调用，其余方法为 0；这证明钩子执行过，不证明控制一定有效。</p>
<p><a href="summary_ai.json">完整统计 JSON</a> · <a href="per_video_ai.csv">逐条 CSV</a> · <a href="annotations_ai.json">AI 标注 JSON</a> · <a href="protocol.json">评审说明</a></p>
<p>离线查看时保留本目录与上一级 videos/ 的相对位置。</p>
<label>方法 <select id="method"><option value="">全部</option>{method_options}</select></label>
<label>动作 <select id="action"><option value="">全部</option>{action_options}</select></label>
<label>状态 <select id="status"><option value="">全部</option><option value="success">完整次数匹配</option><option value="failure">完整次数不匹配</option><option value="uncertain">不确定 / 不可辨认</option></select></label>
<span id="shown">96 条</span>{''.join(cards)}
<script>for(const s of document.querySelectorAll('select'))s.addEventListener('change',()=>{{let n=0;for(const c of document.querySelectorAll('.clip')){{c.hidden=['method','action','status'].some(k=>document.getElementById(k).value&&c.dataset[k]!==document.getElementById(k).value);if(!c.hidden)n++;}}document.getElementById('shown').textContent=n+' 条';}});</script></html>'''
    (OUT/'report_ai.html').write_text(document,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','detail','annotate','summarize'])
    parser.add_argument('--ids',type=int,nargs='+');parser.add_argument('--start',type=int,default=0);parser.add_argument('--end',type=int,default=80)
    parser.add_argument('--crop',type=int,nargs=4)
    parser.add_argument('--items-json')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='detail':detail(args.ids,args.start,args.end,args.crop)
    elif args.command=='annotate':annotate(json.loads(args.items_json) if args.items_json else json.load(sys.stdin))
    else:summarize()
