"""Archive a CPU-only, read-only review of the completed v1/v2 dev experiments.

Does not run inference, change labels/scores, or import manual grades. Small input
files are hashed while loading; model weights and unrelated projects are not read.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb.benchmark.scoring import covered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/zjr/enterprise_kb_agent'))
    args = parser.parse_args()
    base = args.root / 'artifacts/git_support_v1'
    current = base / 'dev_compare_v2_20261028'
    dataset = args.root / 'datasets/git_support_zh_v1'
    archive = current / 'analysis'
    inputs = {}

    def read(path, lines=False):
        data = path.read_bytes()
        inputs[str(path)] = {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest(),
                             'mtime_ns':path.stat().st_mtime_ns}
        return [json.loads(line) for line in data.splitlines() if line] if lines else json.loads(data)

    cases = {r['id']:r for r in read(dataset/'dev.jsonl', True)}
    assert len(cases) == 100 and len({r['family_id'] for r in cases.values()}) == 25
    read(dataset/'freeze.json')
    rows, reports, configs = {}, {}, {}
    for version, dirname in [('v1','dev_compare_20261028'), ('v2','dev_compare_v2_20261028')]:
        for arm in ['fixed-bm25','agent-bm25']:
            name = version + '_' + arm
            folder = base / dirname / arm
            rr = read(folder/'predictions.jsonl', True)
            cfg = read(folder/'config.json')
            reports[name] = read(folder/'report.json')
            status = read(folder/'status.json')
            assert status['status'] == 'complete' and reports[name]['complete']
            assert len(rr) == 100 and len({r['id'] for r in rr}) == 100
            assert set(cfg['case_ids']) == set(cases) == {r['id'] for r in rr}
            assert all(r['family_id'] == cases[r['id']]['family_id'] for r in rr)
            rows[name], configs[name] = rr, cfg
    assert all(c['dataset_receipt'] == configs['v2_agent-bm25']['dataset_receipt'] for c in configs.values())
    assert configs['v2_agent-bm25']['source_fingerprint'] == configs['v2_fixed-bm25']['source_fingerprint']
    comparison = read(current/'comparison/comparison.json')
    queue = read(current/'queue_state.json')
    assert queue['status'] == 'complete'

    docs = read(dataset/'documents.jsonl', True)
    texts = {}
    for doc in docs:
        p = dataset/doc['path']
        data = p.read_bytes()
        assert hashlib.sha256(data).hexdigest() == doc['sha256']
        texts[doc['id']+'_'+doc['title']+'.txt'] = (doc['id'], data.decode())
    for case in read(dataset/'catalog.jsonl', True):
        fixture = case.get('fixture')
        if fixture:
            texts[fixture['id']+'_'+fixture['title']+'.txt'] = (fixture['id'], fixture['text'])

    def chunk_map(arm):
        dbpath = current/arm/'workspace/knowledge.sqlite3'
        connection = sqlite3.connect('file:'+str(dbpath)+'?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        mapping, starts = {}, {}
        try:
            for chunk in connection.execute('SELECT c.*,d.filename FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY d.id,c.ordinal'):
                docid, text = texts[chunk['filename']]
                start = text.find(chunk['text'], starts.get(docid, 0))
                assert start >= 0, (docid, chunk['id'])
                starts[docid] = start + 1
                mapping[chunk['id']] = {'document_id':docid, 'start':start, 'end':start+len(chunk['text'])}
        finally:
            connection.close()
        return mapping

    audits, evidence_summary = [], {}
    for arm in ['fixed-bm25','agent-bm25']:
        mapping = chunk_map(arm)
        rr = rows['v2_'+arm]
        for row in rr:
            case = cases[row['id']]
            source_ids = {e['result']['id'] for e in row['trace'] if e['kind']=='tool' and e['name']=='read_chunk' and 'text' in e['result']}
            spans = [mapping[i] for i in source_ids]
            gold = case['evidence']
            read_full = all(covered(e, spans) >= .95 for e in gold) if gold else None
            cite_full = all(covered(e, row['citations']) >= .95 for e in gold) if gold else None
            audits.append({'id':row['id'], 'arm':arm, 'family_id':row['family_id'], 'category':row['category'],
                'expected_action':case['expected_action'], 'status':row['status'], 'action_correct':row['metrics']['action_correct'],
                'all_facts_diagnostic':row['metrics']['all_required_facts_diagnostic'],
                'read_all_gold':read_full, 'cite_all_gold':cite_full, 'unique_chunks_read':len(source_ids)})
        evaluable = [r for r in audits if r['arm']==arm and r['read_all_gold'] is not None]
        evidence_summary[arm] = {'evaluable':len(evaluable), 'read_all_gold':sum(r['read_all_gold'] for r in evaluable),
            'cite_all_gold':sum(r['cite_all_gold'] for r in evaluable),
            'read_all_but_cite_incomplete':sum(r['read_all_gold'] and not r['cite_all_gold'] for r in evaluable),
            'read_all_but_facts_incomplete':sum(r['read_all_gold'] and not r['all_facts_diagnostic'] for r in evaluable)}

    arm_metrics = {}
    for name, rr in rows.items():
        report = reports[name]
        arm_metrics[name] = {'records':len(rr), 'statuses':dict(Counter(r['status'] for r in rr)),
            'completed':sum(not r['metrics']['failed'] for r in rr), 'action_correct':sum(r['metrics']['action_correct'] for r in rr),
            'facts_all':sum(bool(r['metrics']['all_required_facts_diagnostic']) for r in rr),
            'facts_eligible':sum(r['metrics']['all_required_facts_diagnostic'] is not None for r in rr),
            'model_call_distribution':dict(Counter(r['model_calls'] for r in rr)),
            **{k:report[k] for k in ['model_calls','tool_calls','prompt_tokens','completion_tokens','latency_seconds']},
            'latency_mean_seconds':statistics.mean(r['latency_seconds'] for r in rr)}
    fixed, agent = ({r['id']:r for r in rows['v2_'+a]} for a in ['fixed-bm25','agent-bm25'])
    old_agent = {r['id']:r for r in rows['v1_agent-bm25']}
    wins = [i for i in cases if fixed[i]['metrics']['all_required_facts_diagnostic'] is False and agent[i]['metrics']['all_required_facts_diagnostic'] is True]
    losses = [i for i in cases if fixed[i]['metrics']['all_required_facts_diagnostic'] is True and agent[i]['metrics']['all_required_facts_diagnostic'] is False]
    old_failed = [i for i in cases if old_agent[i]['metrics']['failed']]
    assert all(not agent[i]['metrics']['failed'] for i in old_failed)
    events = [e for r in agent.values() for e in r['trace'] if e['kind']=='tool']
    loop = {'duplicate_events':sum(bool(e['result'].get('duplicate')) for e in events),
        'cases_with_duplicates':sum(any(e['kind']=='tool' and e['result'].get('duplicate') for e in r['trace']) for r in agent.values()),
        'tool_errors':dict(Counter(e['result']['error'] for e in events if 'error' in e['result'])),
        'actual_search_count_distribution':dict(Counter(sum(e['kind']=='tool' and e['name']=='search' and 'results' in e['result'] for e in r['trace']) for r in agent.values()))}

    notes = {
        'unknown-1-2': 'Agent 把公开教程中的示例 URL 当成本公司的生产地址。引用格式合法不等于该引用能证明企业事实；固定 RAG 在这题拒答。',
        'clarify-1-1': '两分支均直接列出删除命令，状态为 answered，未先确认题目缺失的本地/远端目标。',
        'clarify-1-3': '两分支文本提到需要确定目标，但仍以 answered 提交。应结合语义复核区分纯动作标签不一致与实际行为错误，不能把全部动作错判等同事实错误。',
        'branch-management__2-2': 'Agent 回答成查看 testing 与 master 的提交差集，偏离列出分支的意图。固定 RAG 虽命中全部关键词，但把当前分支误当作 master，命令缺少显式 master 参数；关键词通过也可能是错答案。',
        'tagging__1-2': 'Agent 正确解释附注标签能记录打标签者和说明。候选 required_facts 同时要求轻量标签和引用等词，这个变体并未明确要求完整对比；属于标签适配/评分规则需复核案例。',
        'credentials__2-3': 'Agent 使用“不会自动过期”并描述存储文件，评分只接受“不过期/永不”和“磁盘”等字面词，存在同义表达漏判。完整回答是否有其他多余或不准内容仍需独立复核。',
        'hooks__1-3': 'Agent 说明路径、命名和可执行要求，但没有明确去掉示例文件的 .sample 后缀，属于可操作步骤遗漏。'}
    examples = []
    for task_id, note in notes.items():
        c = cases[task_id]
        examples.append({'id':task_id, 'question':c['question'], 'reference_answer':c['reference_answer'],
            'required_facts':c['required_facts'], 'gold_evidence':c['evidence'], 'reviewer_type':'assistant_spot_check_not_independent_grade',
            'review_note':note, 'fixed_answer':fixed[task_id]['answer'], 'agent_answer':agent[task_id]['answer'],
            'fixed_metrics':fixed[task_id]['metrics'], 'agent_metrics':agent[task_id]['metrics']})
    f, a = arm_metrics['v2_fixed-bm25'], arm_metrics['v2_agent-bm25']
    ratios = {k:a[k]/f[k] for k in ['model_calls','tool_calls','prompt_tokens','completion_tokens']}
    ratios['total_tokens']=(a['prompt_tokens']+a['completion_tokens'])/(f['prompt_tokens']+f['completion_tokens'])
    result = {'arms':arm_metrics, 'evidence':evidence_summary, 'loop':loop, 'cost_ratios_agent_over_fixed':ratios,
        'facts_agent_wins':wins, 'facts_agent_losses':losses, 'v1_failed_cases_recovered':old_failed,
        'comparison':comparison, 'limitations':['candidate_labels_not_independently_reviewed','automatic_diagnostics_not_semantic_accuracy',
        '25_correlated_families','shared_gpu_latency','dev_only','combined_policy_change_not_single_factor_ablation']}

    pct = lambda value: f'{value*100:.2f}%'
    lines = ['# 企业知识库 Agent v2 开发集结果分析与存档', '',
        f"分析生成时间：{datetime.now(timezone.utc).isoformat()}。原队列完成时间：{datetime.fromtimestamp(queue['updated_at'],timezone.utc).isoformat()}。", '',
        '## 结论', '',
        '本次完成的是开发集评测。新版解决了这批题目的循环失控与上下文溢出导致的流程失败：旧 Agent 为 62/100 正常完成，新版为 100/100。但本轮没有证据说明多轮 Agent 比固定 RAG 更有价值：关键诊断指标的点估计偏低，模型调用成本显著更高。当前应保留固定 RAG 作为优先基线，Agent 作为可继续验证的受控多步分支；本次未更改在线默认模式。', '',
        '## 数据与比较口径', '',
        '- 40 份公开 Pro Git 中文资料；100 道开发题、25 个题族，每族 4 个相关问法。84 题有标准证据/必要事实要求；其余题目考察拒答、权限或澄清。此语料模拟技术支持，不是企业私有业务资料。',
        '- 固定 RAG 和 Agent 均使用同一 Qwen3-8B、BM25、冻结语料和题目；BF16、8192 上下文、最多 1400 输出 token。Agent 上限仍为 6 次模型调用、12 次工具调用。',
        '- 新版两组各 100 条，ID、题族、数据收据一致，推理源码指纹一致；原始结果、冻结标签和运行代码快照均未修改。旧版数据收据和题目 ID 也相同，但推理代码不同。',
        '- v1→v2 同时调整了提示、工具阶段限制和证据上下文组织，属于整套流程迭代，不是单因素消融。固定 RAG 的本轮重跑也与旧版有小幅波动（全部必要事实命中从 62/84 到 64/84），不能声称推理完全逐位确定。',
        '- 指标分母、题族 bootstrap 沿用原报告。候选标签由代理编写、未独立人工复核；事实关键词命中和引用证据重合都不是语义准确率。冻结的 300 道测试题尚未运行。', '',
        '## 主要指标', '', '| 指标 | 旧 Agent v1 | 本轮固定 RAG | Agent v2 |', '|---|---:|---:|---:|',
        f"| 流程完成 | 62/100 | {f['completed']}/100 | {a['completed']}/100 |",
        f"| 回答/拒答/澄清动作匹配 | 61/100 | {f['action_correct']}/100 | {a['action_correct']}/100 |",
        f"| 必要事实全部命中（诊断） | 43/84（51.19%） | {f['facts_all']}/84（{pct(f['facts_all']/84)}） | {a['facts_all']}/84（{pct(a['facts_all']/84)}） |"]
    for label, key in [('平均必要事实命中（诊断）','required_fact_match_diagnostic'),('引用格式有效','citation_format_valid'),('标准证据引用召回（诊断）','citation_gold_evidence_recall')]:
        vals = [reports[n]['metrics'][key]['mean'] for n in ['v1_agent-bm25','v2_fixed-bm25','v2_agent-bm25']]
        lines.append('| '+label+' | '+' | '.join(pct(v) for v in vals)+' |')
    lines += ['', '38 条旧流程失败均已正常结束，但流程结束不保证事实正确。新版没有发生终止型上下文错误；有一次证据准入被预算保护拒绝，反馈后仍完成了任务。', '',
        f"在 84 道有必要事实要求的题上，新 Agent 相对固定 RAG 有 {len(wins)} 道从不完整变完整、{len(losses)} 道从完整变不完整，净差 {len(wins)-len(losses)} 道。多证据题两者均为 3/8 完整命中，没有观察到该类任务的净增益。", '',
        '## 配对不确定性', '', '| Agent − 固定 RAG | 差值（百分点） | 95% 题族 bootstrap 区间（百分点） |', '|---|---:|---|']
    for key in ['action_correct','all_required_facts_diagnostic','citation_gold_evidence_recall']:
        d=comparison['deltas'][key];lo,hi=d['ci95_family_bootstrap']
        lines.append(f"| {key} | {100*d['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '必要事实和证据召回差异的区间包含 0，不能据此断言总体语义质量显著变差；但点估计和开销均不支持选择当前 Agent 替代固定 RAG。100 题实际只有 25 个题族；多证据、权限、注入、澄清题族更少，局部满分不代表广泛可靠性。', '',
        '## 成本与重复动作', '', '| 项目 | 固定 RAG | Agent v2 | Agent / 固定 RAG |', '|---|---:|---:|---:|']
    for label,key in [('模型调用','model_calls'),('工具调用（含完成提交）','tool_calls'),('输入 tokens','prompt_tokens'),('输出 tokens','completion_tokens')]:
        lines.append(f"| {label} | {f[key]:,} | {a[key]:,} | {ratios[key]:.2f}× |")
    for label,key in [('单题耗时中位数','p50'),('单题耗时 P95','p95')]:
        l,r=f['latency_seconds'][key],a['latency_seconds'][key]
        lines.append(f'| {label} | {l:.2f} s | {r:.2f} s | {r/l:.2f}× |')
    lines += ['', f"输入加输出 token 总量为固定 RAG 的 {ratios['total_tokens']:.2f} 倍。共享 GPU 的延迟仅为本次观察，不能当专用服务吞吐率或严格同算力实验。",
        f"{loop['cases_with_duplicates']}/100 题仍出现重复工具请求，共 {loop['duplicate_events']} 次被去重。90 题用到第 5 次模型调用才完成。真正执行两次不同检索的只有 18 题，一次检索 81 题，零次 1 题。控制器使循环有界，但模型尚未稳定学会何时停止；去重减少重复读取，不会消除已经发生的模型生成成本。", '',
        '## 证据读取与最终引用', '', '| 完整覆盖标准证据的诊断（84 题） | 固定 RAG | Agent v2 |', '|---|---:|---:|']
    for label,key in [('已读原文完整覆盖','read_all_gold'),('最终引用完整覆盖','cite_all_gold'),('已读完整但引用不完整','read_all_but_cite_incomplete'),('已读完整但必要事实未全命中','read_all_but_facts_incomplete')]:
        lines.append(f"| {label} | {evidence_summary['fixed-bm25'][key]} | {evidence_summary['agent-bm25'][key]} |")
    lines += ['', '这里按冻结标准证据片段 95% 覆盖阈值计算，允许多个已读片段联合覆盖。未覆盖标准片段不代表完全没有其他等价证据；完整覆盖也不代表模型理解或引用正确。此诊断用于定位检索、读取、引用选择与表达之间的损失。', '',
        '## 代表性失败与评分局限', '', '以下为本次助手对选定案例的检查，不是独立人工盲评，没有据此改分。完整问题、参考证据、两组答案保存于 case_examples.json。', '']
    for task_id,note in notes.items():
        lines.append(f'- **{task_id}**：{note}')
    lines += ['', '特别是 branch-management 的固定 RAG 虽通过关键词规则，仍有上下文逻辑错误；tagging 与 credentials 则出现合理答案被字面规则扣分的情况。因此表中的 76.19%/71.43% 不能包装为完整问答准确率。', '',
        '## 下一步建议', '',
        '1. 优先补评测可信度：复核问法变体是否共享了过多事实要求，补充关键参数/关系约束与同义表述；对现有两组答案做独立语义评分。任何标签改动生成新数据版本并对两组同时重评分，保留本次冻结结果，不按某一模型答案单独修分。',
        '2. 再迭代受控 Agent：使简单、证据足够的问题早点提交；只有缺少子问题依据时才进行第二次检索，单独检查“示例内容当公司事实”和缺失目标时的澄清。以开发集证据比较策略，不能靠放宽预算或把所有失败改为拒答获得表面完成率。',
        '3. 使用固定 RAG 作为质量/成本基线。只有开发集证明多步策略有价值并冻结方案后，再运行 300 道测试题；本次没有启动新实验、训练或测试集推理。', '',
        '## 存档与复现', '',
        '- 本目录 report.md：人工可读分析；analysis.json：统计；case_audit.jsonl：逐题证据阶段诊断；case_examples.json：选定案例；manifest.json：输入路径、摘要和生成脚本标识。',
        '- 原始预测、工具轨迹、每组报告、源码快照与数据库仍在父目录对应分支中；这里按引用存档，不复制权重或数据集，不覆盖原报告。',
        f'- 原配对报告：[comparison.md]({current}/comparison/comparison.md)。',
        f'- 复现脚本：{Path(__file__).resolve()}。仅 CPU 分析，无模型调用。', '']
    archive.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        (archive/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write('analysis.json',result)
    write('case_examples.json',examples)
    (archive/'case_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits))
    (archive/'report.md').write_text('\n'.join(lines))
    # Verify input predictions/configs/reports were not modified during this read.
    for name,record in inputs.items():
        assert Path(name).stat().st_mtime_ns == record['mtime_ns']
        assert Path(name).stat().st_size == record['bytes']
    write('manifest.json',{'generated_at':datetime.now(timezone.utc).isoformat(), 'inputs':inputs,
          'script':str(Path(__file__).resolve()), 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'outputs':['report.md','analysis.json','case_audit.jsonl','case_examples.json'],
          'inference_calls':0, 'gold_labels_modified':False, 'independent_grading_performed':False})
    print(json.dumps({'report':str(archive/'report.md'),'evidence':evidence_summary,'loop':loop,'wins':wins,'losses':losses,
                      'ratios':ratios},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
