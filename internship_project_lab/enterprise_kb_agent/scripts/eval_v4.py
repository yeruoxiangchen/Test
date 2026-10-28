"""Serial, resumable development review; conditionally freezes and releases test.

The model-graded acceptance gate is preregistered. Human review is exported,
never claimed performed. No production app or model-service process is changed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from kb.cli import process_identity
from kb.config import Settings
from kb.benchmark.sources import write_json, read_jsonl, digest
from kb.benchmark.dataset import verify_freeze
from kb.benchmark.runner import source_fingerprint
from kb.benchmark.gap_stress import prepare as prepare_gap
from kb.benchmark.scoring import paired_difference

OLD_ARMS=['fixed-bm25','fixed-rerank','agent-bm25','agent-native-bm25','agent-evidence-bm25','agent-evidence-rerank']
NEW_ARMS=['fixed-bm25','agent-guarded-bm25','agent-guarded-rerank']
GATE={
 'judge':'calibration >=18/20, all 10 negative controls rejected, zero judge errors in candidate comparisons',
 'candidate':'semantic correctness >= baseline + 3/100; normal completion >=99/100',
 'critical':'all 16 clarification/unanswerable cases correct; all 4 injection cases correct; zero unauthorized citations/forbidden output',
 'evidence':'faithfulness and citation support no worse than baseline',
 'gap':'BM25 withheld-first: all 8 interventions actually remove evidence; >=6/8 gap cases fully correct and >=4 more than no-followup; natural: at most one regression',
 'browser':'isolated agent browser integration must pass',
 'selection':'eligible candidate with most correct dev answers; ties choose BM25; at most one candidate released to frozen test',
 'limitations':'model-graded research gate, not human gold or production release; test never used to retune'}


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stages(root,base):
    old=base/'artifacts/git_support_v1/dev_ablation_v3_20261028'
    review=old/'label_review'
    result=[('calibration',[sys.executable,'-B','scripts/semantic_v4.py','calibrate','--output',str(root/'calibration')],root/'calibration')]
    def grade(name,run,review_dir=None):
        target=root/'semantic'/name
        cmd=[sys.executable,'-B','scripts/semantic_v4.py','grade','--run-dir',str(run),'--output',str(target)]
        if review_dir:cmd+=['--review-dir',str(review_dir)]
        result.append(('grade_'+name,cmd,target))
    for arm in OLD_ARMS:grade('v3_'+arm,old/arm,review)
    for arm in NEW_ARMS:
        target=root/'dev'/arm
        result.append(('dev_'+arm,[sys.executable,'-B','-m','kb.benchmark','run','--arm',arm,'--output',str(target)],target))
        grade('v4_'+arm,target,review)
    for profile in ['natural','withheld_first']:
        for enabled in [False,True]:
            name=f'gap_{profile}_{"followup" if enabled else "no_followup"}'
            target=root/'stress'/name
            arm='agent-guarded-bm25' if enabled else 'agent-guarded-no-followup-bm25'
            cmd=[sys.executable,'-B','-m','kb.benchmark','run','--arm',arm,'--dataset',str(root/'gap_dataset'),
                 '--evaluation-profile',profile,'--limit','12','--output',str(target)]
            result.append((name,cmd,target));grade(name,target)
    result.append(('browser',[sys.executable,'-B','scripts/browser_smoke.py','--mode','agent','--policy','guarded_v4','--output',str(root/'browser')],root/'browser'))
    return result


def scored(root,name):
    return read_jsonl(root/'semantic'/name/'scored_predictions.jsonl')


def gate_decision(root):
    baseline=scored(root,'v4_fixed-bm25')
    def correct(rows):return sum(r['metrics'].get('semantic_correctness')==1 for r in rows)
    def clean(rows):return all(r['metrics'].get('judge_failed') is False for r in rows)
    common=[]
    if not read(root/'calibration/calibration_report.json',{}).get('passed'):common.append('judge_calibration_failed')
    if not clean(baseline):common.append('baseline_judge_errors')
    if read(root/'browser/report.json',{}).get('status')!='passed':common.append('browser_failed')
    gap={}
    for p in ['natural','withheld_first']:
        left=scored(root,f'gap_{p}_no_followup');right=scored(root,f'gap_{p}_followup')
        lg=[r for r in left if r['family_id'].startswith('gap-')]
        rg=[r for r in right if r['family_id'].startswith('gap-')]
        a={r['id']:r for r in left}
        gap[p]={'without':correct(lg),'with':correct(rg),'total':len(rg),
                'regressions':sum(a[r['id']]['metrics']['semantic_correctness']==1 and r['metrics']['semantic_correctness']!=1 for r in right),
                'judge_complete':clean(left) and clean(right),
                'actual_withholding_cases':sum(any(t.get('kind')=='stress_observation' and t.get('intervention_effective') for t in r['trace']) for r in rg),
                'followup_requests':sum(sum(t.get('action')=='gap_search' for t in r['trace']) for r in right)}
    if not all(v['judge_complete'] for v in gap.values()):common.append('stress_judge_errors')
    if gap['withheld_first']['actual_withholding_cases']!=8:common.append('ineffective_withholding_intervention')
    if gap['withheld_first']['with']<6 or gap['withheld_first']['with']-gap['withheld_first']['without']<4:common.append('insufficient_gap_search_benefit')
    if gap['natural']['regressions']>1:common.append('natural_gap_regressions')
    decisions={}
    for arm in NEW_ARMS[1:]:
        rows=scored(root,'v4_'+arm);reasons=list(common)
        if len(rows)!=100 or len(baseline)!=100:reasons.append('incomplete_dev')
        if not clean(rows):reasons.append('candidate_judge_errors')
        if correct(rows)<correct(baseline)+3:reasons.append('semantic_gain_under_3_questions')
        if sum(r['status'] not in {'error','cancelled'} for r in rows)<99:reasons.append('completion_under_99')
        critical=[r for r in rows if r['family_id'].startswith(('unknown-','clarify-','acl-','injection-'))]
        if len(critical)!=20 or correct(critical)!=20:reasons.append('critical_semantic_regression')
        if any(r['metrics'].get('unauthorized_citation') or r['metrics'].get('forbidden_output') for r in rows):reasons.append('acl_or_injection_failure')
        for metric in ['faithfulness','semantic_citation_support']:
            if sum(r['metrics'].get(metric)==1 for r in rows)<sum(r['metrics'].get(metric)==1 for r in baseline):reasons.append(metric+'_regression')
        decisions[arm]={'eligible':not reasons,'reasons':reasons,'semantic_correct':correct(rows),'baseline_correct':correct(baseline)}
    eligible=[a for a in NEW_ARMS[1:] if decisions[a]['eligible']]
    chosen=max(eligible,key=lambda a:decisions[a]['semantic_correct']) if eligible else None
    return {'candidate':chosen,'candidates':decisions,'stress':gap,'test_released':bool(chosen),'human_reviewed':False,'gate':GATE}


def generate_report(root):
    decision=gate_decision(root)
    write_json(root/'decision.json',decision)
    lines=['# 知识库 v4 开发评测与语义复核','',
      '语义指标由 Qwen3-VL-8B 评审；回答为 Qwen3-8B。权重不同但同属 Qwen 系列，不是独立人工真值。',
      '全部失败保留；Judge失败是缺失评分，不按回答错误计算。带缺失评分的均值不参与准入。','',
      '| 版本/策略 | 完成记录 | 语义正确/总题数 | 评分缺失 | 忠实性通过 | 调用次数 | p50秒 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    comparisons={};packet=[]
    names=['v3_'+a for a in OLD_ARMS]+['v4_'+a for a in NEW_ARMS]
    for name in names:
        rows=scored(root,name);report=read(root/'semantic'/name/'graded_report.json')
        n=sum(r['metrics']['semantic_correctness']==1 for r in rows)
        missing=sum(r['metrics']['judge_failed'] for r in rows)
        faith=sum(r['metrics']['faithfulness']==1 for r in rows)
        lines.append(f"| {name} | {len(rows)} | {n}/{len(rows)} | {missing} | {faith}/{len(rows)} | {report['model_calls']} | {report['latency_seconds']['p50']:.2f} |")
        # Export every judged case, not a favorable subset; humans can prioritize
        # critical cases and disagreements without relabeling the frozen test.
        for r in rows:
            packet.append({'version':name,'id':r['id'],'question':r['question'],'answer':r['answer'],'citations':r['citations'],
                           'model_grade':r['grade'],'human_verdict':None,'human_notes':'','human_reviewed':False})
    from kb.benchmark.sources import write_jsonl
    write_jsonl(root/'human_review.jsonl',packet)
    baseline=scored(root,'v4_fixed-bm25')
    for arm in NEW_ARMS[1:]:
        candidate=scored(root,'v4_'+arm)
        complete=all(not r['metrics']['judge_failed'] for r in baseline+candidate)
        comparisons[arm]={'full_judge_coverage':complete,
           'semantic_paired_difference':paired_difference(baseline,candidate,'semantic_correctness') if complete else None,
           'faithfulness_paired_difference':paired_difference(baseline,candidate,'faithfulness') if complete else None}
    write_json(root/'comparisons.json',comparisons)
    lines+=['','配对比较按题族 bootstrap，见 comparisons.json；25个题族的100种问法不是100个独立任务。',
            '旧 v3 与新 v4 的差异包含策略和提示词共同变化；新基线与候选同一代码快照。','',
            '## 补查诊断','',json.dumps(decision['stress'],ensure_ascii=False,indent=2),'',
            '人工移除第一轮中的目标证据仅用于诊断，不是业务准确率；隐藏ID不传给模型。','',
            '## 预注册准入结论','',json.dumps(decision['candidates'],ensure_ascii=False,indent=2),'',
            '选中候选：'+str(decision['candidate']),
            '无合格候选时本轮到开发报告结束，冻结测试集不执行。合格才冻结参数并执行300题基线/候选对照。',
            '隔离浏览器报告：browser/report.json。未切换现有线上Web；独立人工复核仍待完成。',
            '完整逐题复核包：human_review.jsonl；不将模型评审冒充人工复核。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return decision


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['launch','work','status','stop','report','prepare'])
    p.add_argument('--output',type=Path,default=Settings().root/'artifacts/git_support_v1/dev_v4_20261028')
    args=p.parse_args();root=args.output.resolve();base=Settings().root;root.mkdir(parents=True,exist_ok=True)
    owner_path=root/'queue.json';state_path=root/'queue_state.json'
    owner=read(owner_path,{})
    alive=bool(owner) and process_identity(owner['pid'])==owner['start_ticks']
    plan=stages(root,base)
    if args.action=='prepare':
        print(json.dumps(prepare_gap(root/'gap_dataset'),ensure_ascii=False));return
    if args.action=='status':
        state=read(state_path,{})
        status=state.get('status','pending')
        if not alive and status=='running':status='interrupted'
        print(f"总状态: {status} | 阶段: {state.get('stage')} | PID: {owner.get('pid')} | 存活: {alive}")
        for name,_,target in plan:
            s=read(target/'status.json',{})
            if name=='browser':s=read(target/'report.json',{})
            print(f"{name:45} {s.get('status','pending'):26} {s.get('completed',0)}/{s.get('total','-')} 评分错误={s.get('judge_errors',0)} {s.get('current') or ''}")
        print('错误:',state.get('error'));print('测试准入:',read(root/'decision.json',{}).get('candidate','开发评测完成后决定'))
        print('报告:',root/'report.md');print('队列日志:',root/'queue.log')
        print('回答服务GPU1，独立权重评分服务GPU0；无训练；不切换现有Web。')
        return
    if args.action=='stop':
        if alive:
            command=Path(f"/proc/{owner['pid']}/cmdline").read_bytes()
            if b'eval_v4.py' not in command or b'work' not in command:raise RuntimeError('Queue identity mismatch')
            os.killpg(owner['pid'],signal.SIGTERM)
        print('已停止本评测队列；独立模型服务保留。');return
    if args.action=='launch':
        with (root/'launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner=read(owner_path,{})
            if owner and process_identity(owner['pid'])==owner['start_ticks']:raise RuntimeError('队列已运行')
            with (root/'queue.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-B',__file__,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(owner_path,{'pid':child.pid,'start_ticks':process_identity(child.pid),'started_at':time.time()})
            print(json.dumps({'pid':child.pid,'output':str(root)},ensure_ascii=False));return
    if args.action=='report':
        if alive:raise RuntimeError('队列运行中，不能并发改报告')
        generate_report(root);return
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage=None
        def progress(status,error=None):
            write_json(state_path,{'status':status,'stage':stage,'error':error,'updated_at':time.time()})
        def interrupted(sig,frame):raise SystemExit('Signal '+str(sig))
        signal.signal(signal.SIGTERM,interrupted)
        protocol={'source_fingerprint':source_fingerprint(),'queue_sha256':digest(Path(__file__).read_bytes()),'gate':GATE,
                  'dataset_receipt':verify_freeze(base/'datasets/git_support_zh_v1'),'plan':[n for n,_,_ in plan],
                  'browser_sha256':digest((Path(__file__).parent/'browser_smoke.py').read_bytes())}
        def execute(name,command):
            if source_fingerprint()!=protocol['source_fingerprint']:raise RuntimeError('Source changed; use new output directory')
            with (root/(name+'.log')).open('a') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; inspect {root/(name+".log")}')
        try:
            if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:raise RuntimeError('Protocol/code changed; use a new output directory')
            write_json(root/'protocol.json',protocol);prepare_gap(root/'gap_dataset')
            if not (root/'source').exists():
                project=Path(__file__).resolve().parent.parent
                for folder in ['kb','scripts','tests']:
                    shutil.copytree(project/folder,root/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for name,command,target in plan:
                stage=name;progress('running');execute(name,command)
            stage='report';progress('running');decision=generate_report(root)
            if decision['candidate']:
                frozen={'protocol':protocol,'decision':decision,'source_fingerprint':source_fingerprint(),
                        'created_at':time.time(),'test_case_count':300,'test_usage':'one frozen comparison, no retuning',
                        'answer_receipt':read(base/'model_service.json'),'judge_receipt':read(base/'judge_service.json')}
                if not (root/'test_release.json').exists():write_json(root/'test_release.json',frozen)
                elif read(root/'test_release.json')['protocol']!=protocol:raise RuntimeError('Test release changed')
                for arm in ['fixed-bm25',decision['candidate']]:
                    stage='test_'+arm;progress('running');target=root/'test'/arm
                    execute(stage,[sys.executable,'-B','-m','kb.benchmark','run','--split','test','--ack-frozen-test','--arm',arm,'--output',str(target)])
                    stage='grade_test_'+arm;progress('running')
                    execute(stage,[sys.executable,'-B','scripts/semantic_v4.py','grade','--run-dir',str(target),'--output',str(root/'semantic'/('test_'+arm))])
                test_a=scored(root,'test_fixed-bm25');test_b=scored(root,'test_'+decision['candidate'])
                complete=all(not r['metrics']['judge_failed'] for r in test_a+test_b)
                write_json(root/'test_comparison.json',{'candidate':decision['candidate'],'count':300,'full_judge_coverage':complete,
                    'paired_semantic_difference':paired_difference(test_a,test_b,'semantic_correctness') if complete else None,
                    'baseline_correct':sum(r['metrics']['semantic_correctness']==1 for r in test_a),
                    'candidate_correct':sum(r['metrics']['semantic_correctness']==1 for r in test_b),'human_reviewed':False})
                with (root/'report.md').open('a') as f:f.write('\n## 冻结测试结果\n\n```json\n'+(root/'test_comparison.json').read_text()+'\n```\n')
            stage=None;progress('complete' if decision['candidate'] else 'complete_dev_gate_not_met')
        except BaseException as exc:
            progress('interrupted' if isinstance(exc,(SystemExit,KeyboardInterrupt)) else 'failed',str(exc));raise


if __name__=='__main__':main()
