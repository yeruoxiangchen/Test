"""Archive completed v3 diagnostics, original/reviewed scores, and known failures.

CPU-only analysis of saved outputs. No inference, online policy changes, or gold edits.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb.benchmark.scoring import covered, paired_difference

ARMS = ['fixed-bm25', 'fixed-rerank', 'agent-bm25', 'agent-native-bm25',
        'agent-evidence-bm25', 'agent-evidence-rerank']
LABELS = ['固定RAG', '固定RAG＋重排', '原有多轮Agent', '原生工具消息Agent', '先取证Agent', '先取证Agent＋重排']
PAIRS = {'retrieval': ('retrieval-bm25', 'retrieval-rerank'), 'rerank': ('fixed-bm25', 'fixed-rerank'),
         'messages': ('agent-bm25', 'agent-native-bm25'), 'workflow': ('fixed-bm25', 'agent-evidence-bm25'),
         'workflow_rerank': ('fixed-rerank', 'agent-evidence-rerank'),
         'evidence_rerank': ('agent-evidence-bm25', 'agent-evidence-rerank')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/zjr/enterprise_kb_agent'))
    args = parser.parse_args()
    suite = args.root/'artifacts/git_support_v1/dev_ablation_v3_20261028'
    dataset = args.root/'datasets/git_support_zh_v1'
    output = suite/'analysis'
    inputs = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'mtime_ns': path.stat().st_mtime_ns}
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

    recovery = read(suite/'reporting_manifest.json')
    assert recovery['original_inputs_unchanged']
    status = read(suite/'queue_state.json')
    assert status['status'] == 'complete'
    protocol = read(suite/'protocol.json')
    cases = {c['id']: c for c in read(dataset/'dev.jsonl', True)}
    assert len(cases) == 100 and len({c['family_id'] for c in cases.values()}) == 25
    original, reviewed, reports = {}, {}, {}
    for arm in protocol['arms']:
        cfg = read(suite/arm/'config.json')
        rows = read(suite/arm/'predictions.jsonl', True)
        assert cfg['source_fingerprint'] == protocol['source_fingerprint']
        assert len(rows) == 100 and {r['id'] for r in rows} == set(cases)
        assert all(r['family_id'] == cases[r['id']]['family_id'] for r in rows)
        original[arm] = {r['id']: r for r in rows}
        reports[arm] = read(suite/arm/'report.json')
        if arm in ARMS:
            reviewed[arm] = {r['id']: r for r in read(suite/'label_review'/arm/'rescored_predictions.jsonl', True)}
            assert set(reviewed[arm]) == set(cases)
    comparisons = {name: read(suite/'comparison'/name/'comparison.json') for name in PAIRS}

    texts = {}
    for doc in read(dataset/'documents.jsonl', True):
        raw = (dataset/doc['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == doc['sha256']
        texts[doc['id']+'_'+doc['title']+'.txt'] = (doc['id'], raw.decode())
    for c in read(dataset/'catalog.jsonl', True):
        f = c.get('fixture')
        if f:
            texts[f['id']+'_'+f['title']+'.txt'] = (f['id'], f['text'])

    metrics, audits = {}, []
    for arm in ARMS:
        mapping, starts = {}, {}
        database = suite/arm/'workspace/knowledge.sqlite3'
        con = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            for chunk in con.execute('SELECT c.*,d.filename FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY d.id,c.ordinal'):
                doc, text = texts[chunk['filename']]
                start = text.find(chunk['text'], starts.get(doc, 0))
                assert start >= 0
                mapping[chunk['id']] = {'document_id':doc, 'start':start, 'end':start+len(chunk['text'])}
                starts[doc] = start+1
        finally:
            con.close()
        rows = list(original[arm].values())
        events = [e for r in rows for e in r['trace'] if e['kind'] == 'tool']
        metrics[arm] = {'statuses': dict(Counter(r['status'] for r in rows)),
            'facts_original': sum(bool(r['metrics']['all_required_facts_diagnostic']) for r in rows),
            'facts_reviewed': sum(bool(r['metrics']['all_required_facts_diagnostic']) for r in reviewed[arm].values()),
            'fact_denominator': sum(r['metrics']['all_required_facts_diagnostic'] is not None for r in rows),
            'action_correct': sum(r['metrics']['action_correct'] for r in rows),
            'duplicate_events': sum(bool(e.get('result', {}).get('duplicate')) for e in events),
            'cases_with_duplicates': sum(any(e['kind']=='tool' and e.get('result', {}).get('duplicate') for e in r['trace']) for r in rows),
            'gap_search_requests': sum(e['kind']=='policy' and e.get('action')=='gap_search' for r in rows for e in r['trace']),
            'model_call_distribution': dict(Counter(r['model_calls'] for r in rows)),
            'execution_errors': [{'id':r['id'], 'error':r['error']} for r in rows if r['status']=='error'],
            **{k: reports[arm][k] for k in ('model_calls', 'prompt_tokens', 'completion_tokens', 'latency_seconds')}}
        arm_audits = []
        for row in rows:
            gold = cases[row['id']]['evidence']
            ids = {e['result']['id'] for e in row['trace'] if e['kind']=='tool' and e['name']=='read_chunk' and 'text' in e['result']}
            spans = [mapping[i] for i in ids]
            audit = {'id':row['id'], 'arm':arm, 'status':row['status'], 'sources_read':len(ids),
                'read_all_gold': all(covered(g, spans)>=.95 for g in gold) if gold else None,
                'cite_all_gold': all(covered(g, row['citations'])>=.95 for g in gold) if gold else None,
                'facts_original': row['metrics']['all_required_facts_diagnostic'],
                'facts_reviewed': reviewed[arm][row['id']]['metrics']['all_required_facts_diagnostic']}
            arm_audits.append(audit)
        audits += arm_audits
        eligible = [a for a in arm_audits if a['read_all_gold'] is not None]
        metrics[arm]['evidence_audit'] = {'eligible':len(eligible), 'read_all_gold':sum(a['read_all_gold'] for a in eligible),
            'cite_all_gold':sum(a['cite_all_gold'] for a in eligible),
            'read_all_but_facts_incomplete':sum(a['read_all_gold'] and not a['facts_reviewed'] for a in eligible)}

    reviewed_deltas = {}
    for name, (left, right) in PAIRS.items():
        if name == 'retrieval':
            continue
        a, b = reviewed[left], reviewed[right]
        reviewed_deltas[name] = {'delta': paired_difference(list(a.values()), list(b.values()), 'all_required_facts_diagnostic'),
            'wins': [i for i in cases if a[i]['metrics']['all_required_facts_diagnostic'] is False and b[i]['metrics']['all_required_facts_diagnostic'] is True],
            'losses': [i for i in cases if a[i]['metrics']['all_required_facts_diagnostic'] is True and b[i]['metrics']['all_required_facts_diagnostic'] is False]}

    notes = {
        'unknown-1-2': '原有和原生工具消息Agent仍把教程示例URL当成公司的真实URL。先取证两组均拒答。',
        'unknown-1-1': '原生工具消息Agent达到6次调用上限未完成；该失败保留在100题分母中。',
        'unknown-1-3': '固定RAG＋重排因为行内引用和citations不一致而失败；没有隐藏重试，也未删除这条记录。',
        'clarify-1-3': '先取证＋重排虽然提交clarify，正文仍建议目标未定时先尝试删除本地标签，再建议确认位置。动作标签正确不能证明澄清行为正确。',
        'branch-management__2-2': '固定BM25/先取证BM25命中关键词却把testing误当作master，缺少显式master参数。先取证＋重排给出了正确的git branch --no-merged master，但也加入了提交差集命令。需语义审查条件与操作对象，不能只数词。',
    }
    examples = [{'id':i, 'question':cases[i]['question'], 'reference_answer':cases[i]['reference_answer'],
        'gold_evidence':cases[i]['evidence'], 'reviewer_type':'assistant_spot_check_not_independent_grade', 'note':note,
        'answers':{a:{k:original[a][i].get(k) for k in ('status','answer','error','metrics')} for a in ARMS}}
        for i, note in notes.items()]

    pct = lambda value: f'{100*value:.2f}%'
    lines = ['# 知识库 v3 开发集结果分析', '',
        '8个分支的800条记录均已产生；2个检索分支各100条，6个问答分支各100条，其中2条问答流程失败保留在各自分母。原队列失败仅因为汇总代码将 evaluable 元信息当成成绩指标。现已修复并完成CPU汇总，没有重新生成答案。', '',
        '## 比较口径', '',
        '- 同一Qwen3-8B、语料、100题开发集、25个题族；84题有标准证据/事实要求。300题冻结测试集未运行。',
        '- 原始标签与修订标签并列报告。修订标签是查看v2错误后的助手复核，不是独立人工标注；关键词命中与引用重合均不等于语义准确率。',
        '- 六组配对报告保留题族bootstrap区间。工具消息对照也改变了上一轮参数呈现；新流程包含提示与策略变化，不宣称纯单变量的架构收益。',
        '- 原始protocol/config/predictions/report/评审输入均校验未改变，记录于 ../reporting_manifest.json；推理与汇总代码指纹分开。', '',
        '## 六组问答结果', '',
        '| 分支 | 正常结束 | 动作匹配 | 原标签全事实诊断 | 修订标签全事实诊断 | 引用证据召回 | 模型调用 | p50秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        m = metrics[arm]
        lines.append(f"| {label} | {100-len(m['execution_errors'])}/100 | {m['action_correct']}/100 | {m['facts_original']}/84（{pct(m['facts_original']/84)}） | {m['facts_reviewed']}/84（{pct(m['facts_reviewed']/84)}） | {pct(reports[arm]['metrics']['citation_gold_evidence_recall']['mean'])} | {m['model_calls']} | {m['latency_seconds']['p50']:.2f} |")
    lines += ['', '## 哪些改动有价值', '',
        '1. **重排提高了证据的排序质量。** Recall@1从43.45%升至60.71%，差值+17.26个百分点，95%题族bootstrap区间[+4.17,+29.76]；Recall@5从83.33%升至91.67%，差值+8.33个百分点，区间[0.00,+19.05]。前5片段完整覆盖标准证据的题数从67/84增至73/84。收益首先体现在检索和引用，而非已经证明答案准确率提高。',
        '2. **原生工具消息没有解决多轮Agent的主要问题。** 调用488→419（减少14.14%），但原标签事实诊断60/84→56/84，修订后66/84→61/84；重复工具请求239→221，仍有一题到达调用上限。事实差值区间包含0；现有证据不支持直接替换原分支，也不能把消息格式定为唯一根因。',
        '3. **先取证流程明显降低相对多轮Agent的调用成本。** 不重排版本100题只用101次模型调用，低于原有488次；相对固定RAG的100次则没有成本优势。重排版本恰好100次。它更接近一条受控RAG流程。',
        '4. **本次没有验证补查收益。** 两个先取证分支的 gap_search 请求均为0；不重排版本多出的一次调用用于引用格式修正。不能将这轮结果包装成多步检索/缺口补查有效。',
        '5. **先取证＋重排的动作匹配与引用最好，但事实完整性未胜出。** 动作100/100，拒答12/12、clarify状态4/4；引用证据召回83.33%，相对不重排版本72.62%提高10.71个百分点，区间[+0.60,+22.62]。但原标签事实仍62/84；修订后65/84，低于固定BM25的69/84。',
        '6. **CPU重排增加了可见延迟。** 单检索p50约0.016→11.43秒，固定问答5.27→16.40秒，新流程＋重排17.01秒。模型调用更少不意味着端到端更快；这是共享服务器观测，并非专用吞吐测试。', '',
        '## 证据链诊断', '', '| 分支 | 已读覆盖全部标准证据/84 | 最终引用覆盖全部标准证据/84 | 已读齐但修订事实未全命中 |', '|---|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        e = metrics[arm]['evidence_audit']
        lines.append(f"| {label} | {e['read_all_gold']} | {e['cite_all_gold']} | {e['read_all_but_facts_incomplete']} |")
    lines += ['', '覆盖按冻结证据95%字符区间计算，可由多片段联合覆盖；没有覆盖不等于没有其他等价证据，覆盖全部也不证明正确理解。', '',
        '## 原标签配对差异', '', '| 对照（右−左） | 全事实诊断差值/百分点 | 95%区间 | 引用召回差值/百分点 | 95%区间 |', '|---|---:|---|---:|---|']
    for name in PAIRS:
        if name == 'retrieval':
            continue
        ds = comparisons[name]['deltas']
        values = []
        for key in ('all_required_facts_diagnostic', 'citation_gold_evidence_recall'):
            d = ds[key]; lo, hi = d['ci95_family_bootstrap']
            values += [f"{100*d['mean']:+.2f}", f"[{100*lo:+.2f}, {100*hi:+.2f}]"]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '修订标签配对区间、逐题改善/退步列表见 analysis.json。小样本、多重开发对照尚未做多重比较校正；不把某个区间解读为广泛业务效果的证明。', '',
        '## 仍需处理的真实错误', '', '以下为助手检查，不是独立盲评，也没有据此修改成绩。']
    lines += [f'- **{i}**：{note}' for i, note in notes.items()]
    lines += ['', '## 下一步', '',
        '优先对已保存答案做独立语义评分和人工校准，特别检查命令参数、当前分支、教程示例与企业事实、缺失条件时是否先澄清。无需先重跑全部推理。',
        '保留固定BM25作为低成本基线，先取证＋重排作为待验证候选。针对确实缺证据的任务验证补查触发率、补查后的改善/损害和成本；当前0次触发不支持继续强调多轮收益。CPU重排延迟优化另做性能对照，不与语义改动混在一起。',
        '本次只修复汇总并分析，未切换网页策略、启动新训练或消费测试集。独立Judge尚未配置，不能把状态complete误读为独立语义评审已完成。', '',
        '## 存档', '',
        '- report.md：本分析；analysis.json：指标、轨迹、原/修订标签配对结果；case_audit.jsonl：600条证据链检查；case_examples.json：5个案例的六组答案；manifest.json：输入摘要。',
        '- 父目录 report.md 为自动总表，comparison/ 保存六组配对，label_review/ 保存独立版本的标签诊断与评审包。',
        f'- CPU复现脚本：{Path(__file__).resolve()}。', '']

    output.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        (output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    write('analysis.json', {'metrics':metrics, 'comparisons':comparisons, 'reviewed_deltas':reviewed_deltas,
          'limitations':['dev_only', 'candidate_labels', 'not_independent_semantic_grades', 'shared_server_latency', 'no_gap_search_triggered']})
    write('case_examples.json', examples)
    (output/'case_audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in audits))
    (output/'report.md').write_text('\n'.join(lines))
    for path, receipt in inputs.items():
        assert Path(path).stat().st_mtime_ns == receipt['mtime_ns'] and Path(path).stat().st_size == receipt['bytes']
    write('manifest.json', {'generated_at':datetime.now(timezone.utc).isoformat(), 'inputs':inputs,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'inference_calls':0, 'original_inputs_modified':False, 'independent_grading_performed':False})
    print(json.dumps({'report':str(output/'report.md'), 'audit_records':len(audits),
          'evidence':{a:m['evidence_audit'] for a,m in metrics.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
