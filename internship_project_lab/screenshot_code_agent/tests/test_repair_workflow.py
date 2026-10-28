import ast
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_harness_compare import old_input as inherited_old_input

from screenshot_agent.config import ImageInput, Task
from screenshot_agent.control import BudgetExceeded, Events, RunCancelled, atomic_json
from screenshot_agent.hidden_eval import grade_hidden, prepare_hidden_task
from screenshot_agent.repo_search import RepoSearch
from screenshot_agent.worker import anchored_patch
from screenshot_agent.workflow import METHOD, RepairWorkflow, run_workflow_task
from screenshot_agent.workflow_config import WorkflowBudget, load_workflow_config
from screenshot_agent.workflow_demo import PLAN, DemoAPI, demo, patch
from screenshot_agent.workflow_view import export_delivery, render_run

ROOT = Path(__file__).resolve().parents[1]
old_input = inherited_old_input


def config():
    return load_workflow_config(ROOT/'configs/workflow-local.json')


def basic_task(tmp_path):
    repo = tmp_path/'input'
    repo.mkdir()
    (repo/'maths.py').write_text('def add(a, b):\n    return a-b\n')
    (repo/'tests').mkdir()
    (repo/'tests/check.py').write_text('immutable original test\n')
    return Task(task_id='addition',repo=str(repo),problem='Fix add to return the sum',
                test_command="python3 -c 'from maths import add; assert add(2, 3) == 5'",protected_paths=['tests'])


def test_retrieval_ranks_symbol_and_updates_changed_deleted_files(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'prices.py').write_text('def discounted_price(price, percent):\n    return price - percent\n')
    (repo/'noise.py').write_text('def greet(name):\n    return name\n')
    index = RepoSearch(repo,tmp_path/'index.json')
    assert index.search('discounted_price percent')[0]['path']=='prices.py'
    assert 'discounted_price :1' in index.repo_map()['output']
    assert index.refresh()['updated_files']==0
    (repo/'prices.py').write_text('def renamed_total():\n    return 2\n')
    (repo/'noise.py').unlink()
    result = index.search('renamed_total')
    assert result[0]['symbols'][0]['name']=='renamed_total'
    assert 'noise.py' not in index.files


def test_index_excludes_external_links_large_files_and_reports_partial_coverage(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'.git').mkdir()
    (repo/'.git/private.py').write_text('PRIVATE_SECRET')
    outside = tmp_path/'outside.py'
    outside.write_text('PRIVATE_SECRET')
    (repo/'linked.py').symlink_to(outside)
    (repo/'big.py').write_text('secret'*500)
    (repo/'ok.py').write_text('def target():\n    return 1\n')
    index = RepoSearch(repo,tmp_path/'index.json',max_file_bytes=100)
    assert index.search('PRIVATE_SECRET')==[]
    assert set(index.files)=={'ok.py'}
    assert index.last['skipped_large_files']==1
    limited = RepoSearch(repo,tmp_path/'limited.json',max_chunks=1)
    assert limited.repo_map()['truncated']


def test_multiple_patches_match_original_spans_and_reject_ambiguous_or_overlapping():
    assert anchored_patch('a = 1\nb = 2\n',patch('a = 1','b = 2')+'\n'+patch('b = 2','c = 3'))=='b = 2\nc = 3\n'
    with pytest.raises(ValueError,match='exactly once'):
        anchored_patch('aa aa',patch('aa','bb'))
    with pytest.raises(ValueError,match='overlap'):
        anchored_patch('abcde',patch('abc','x')+'\n'+patch('cde','y'))
    ast.parse((ROOT/'src/screenshot_agent/worker.py').read_text(),feature_version=(3,6))


@pytest.mark.sandbox
def test_patch_tool_no_partial_mutation_and_readonly_tests(sandbox):
    before = (sandbox.repo/'sample.py').read_text()
    result = json.loads(sandbox.filesystem('apply_patch',{'file_path':'sample.py',
        'patch':patch('value = 1','value = 2')+'\n'+patch('missing','present')},feedback_version=2))
    assert result['returncode']!=0
    assert (sandbox.repo/'sample.py').read_text()==before
    result = json.loads(sandbox.filesystem('apply_patch',{'file_path':'tests/test_fixed.py',
        'patch':patch('immutable','changed')},feedback_version=2))
    assert result['returncode']!=0
    assert (sandbox.repo/'tests/test_fixed.py').read_text()=='immutable\n'


def test_resume_preserves_usage_and_charges_inflight_request(tmp_path):
    cfg = config()
    budget = WorkflowBudget(cfg,Events(tmp_path/'events'),lambda m,t:100,tmp_path/'budget.json')
    budget.reserve('code',cfg.code,[])
    budget.tool('read_file',{})
    restored = WorkflowBudget(cfg,Events(tmp_path/'events2'),lambda m,t:100,tmp_path/'budget.json',resume=True)
    assert restored.code_calls==1 and restored.tool_calls==1
    assert restored.accounted_prompt==100 and restored.accounted_completion==cfg.code.max_output_tokens
    assert restored.estimated_calls==1 and restored.pending_tokens is None
    cfg = cfg.model_copy(update={'total_prompt_tokens':100})
    restored.config=cfg
    with pytest.raises(BudgetExceeded,match='prompt token'):
        restored.reserve('vision',cfg.vision,[])


def test_resume_reads_remaining_screenshots_without_repeating_first(tmp_path):
    task = basic_task(tmp_path).model_copy(update={'images':[
        ImageInput(image_id='first',path='unused.png'),ImageInput(image_id='second',path='unused.png')]})
    flow = RepairWorkflow(task,config(),tmp_path)
    flow.state = {'readings':[{'image_id':'first','raw_text':'already read'}]}
    flow.budget = WorkflowBudget(config(),flow.events,lambda m,t:100,tmp_path/'budget.json')
    seen = []
    flow.reader = SimpleNamespace(read=lambda image_id:seen.append(image_id) or SimpleNamespace(model_dump=lambda:{'raw_text':'second'}))
    def stop_after_reading():
        raise RunCancelled('test ends after reading')
    flow.refresh_index = stop_after_reading
    with pytest.raises(RunCancelled):
        flow.initialize_agent()
    assert seen==['second']
    assert [r['image_id'] for r in flow.state['readings']]==['first','second']


@pytest.mark.sandbox
def test_complete_demo_executes_failure_review_retry_and_clean_verification(tmp_path):
    data = demo(tmp_path)
    result = data['result']
    assert result['status']=='verified',result
    assert result['retry_count']==1
    assert result['changed_files']==['pricing.py']
    assert result['model_inference'] is False and data['gpu_started'] is False
    folder = Path(data['report']).parent
    state = json.loads((folder/'workflow.json').read_text())
    assert [c['passed'] for c in state['checks']]==[False,False,True]
    assert state['checks'][0]['patch_bytes']==0  # Agent's reproduction-time run_tests is now recorded too.
    assert state['checks'][-1]['reproduction']['status']=='verified'
    assert 'AssertionError' in state['readings'][0]['raw_text']
    assert 'error_screen.png' in (folder/'report.html').read_text()
    assert '不代表模型修复成绩' in (folder/'report.html').read_text()
    assert 'return price - percent' in (folder.parent/'input/repo/pricing.py').read_text()
    assert 'private' not in zipfile.ZipFile(data['delivery']).namelist()


@pytest.mark.sandbox
def test_model_claim_done_does_not_count_as_success(tmp_path,monkeypatch):
    cfg = config().model_copy(update={'workflow':config().workflow.model_copy(update={'max_retries':0})})
    monkeypatch.setattr(DemoAPI,'script',[PLAN,'done','done','done'])
    result = run_workflow_task(basic_task(tmp_path),cfg,tmp_path/'run',reader_name='ocr',
                              api_factory=DemoAPI,counter=lambda m,t:100,model_inference=False)
    assert not result['resolved'] and result['patch_bytes']==0 and result['status']=='unresolved'


@pytest.mark.sandbox
def test_interrupted_workspace_resumes_without_replaying_mutation(tmp_path,monkeypatch):
    task,cfg = basic_task(tmp_path),config()
    monkeypatch.setattr(DemoAPI,'script',[PLAN,'located','reproduced',
        ('apply_patch',{'file_path':'maths.py','patch':patch('return a-b','return a+b')}),RunCancelled('test interruption')])
    folder = tmp_path/'run'
    first = run_workflow_task(task,cfg,folder,reader_name='ocr',api_factory=DemoAPI,counter=lambda m,t:100,model_inference=False)
    assert first['status']=='paused',first
    assert 'return a+b' in (folder/'workspace/maths.py').read_text()
    calls = first['usage']['code_model_calls']
    monkeypatch.setattr(DemoAPI,'script',['Current patch already implements addition.'])
    second = run_workflow_task(task,cfg,folder,reader_name='ocr',resume=True,api_factory=DemoAPI,counter=lambda m,t:100)
    assert second['status']=='verified',second
    assert second['usage']['code_model_calls']==calls+1
    assert second['model_inference'] is False
    with pytest.raises(ValueError,match='already ended'):
        run_workflow_task(task,cfg,folder,reader_name='ocr',resume=True)


@pytest.mark.sandbox
def test_reference_grading_never_becomes_retry_feedback(old_input,tmp_path,monkeypatch):
    clean = tmp_path/'hidden'
    prepare_hidden_task(old_input/'task.json',clean)
    task = Task.from_file(clean/'task.json')
    cfg = config().model_copy(update={'workflow':config().workflow.model_copy(update={'max_retries':0})})
    seen = []
    class HiddenAPI(DemoAPI):
        def complete(self,messages,*args,**kwargs):
            text = json.dumps(messages)
            assert not any(s in text for s in ('PRIVATE_GOLD','SECRET_REFERENCE_TEST','928617'))
            seen.append(text)
            return super().complete(messages,*args,**kwargs)
    monkeypatch.setattr(HiddenAPI,'script',[PLAN,'located','checked',
        ('apply_patch',{'file_path':'miniapp.py','patch':patch('return a-b','return a+b')}),'done'])
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade',lambda metadata,log:{'resolved':'hidden-ok' in log.read_text()})
    folder = tmp_path/'run'
    def grade(*args,**kwargs):
        assert json.loads((folder/'workflow.json').read_text())['grading_started']
        assert len(seen)==5
        assert not (folder/'workspace/hidden_spec.py').exists()
        return grade_hidden(*args,**kwargs)
    result = run_workflow_task(task,cfg,folder,reader_name='ocr',api_factory=HiddenAPI,grader=grade,counter=lambda m,t:100)
    assert result['status']=='verified',result
    assert len(seen)==5


@pytest.mark.sandbox
def test_resume_after_grading_started_never_runs_agent(tmp_path,monkeypatch):
    task,cfg = basic_task(tmp_path),config()
    monkeypatch.setattr(DemoAPI,'script',[PLAN,'located','checked',
        ('apply_patch',{'file_path':'maths.py','patch':patch('return a-b','return a+b')}),'done'])
    original = RepairWorkflow.finalize
    def interrupt(self):
        self.save(grading_started=True)
        raise RunCancelled('interrupted grader')
    monkeypatch.setattr(RepairWorkflow,'finalize',interrupt)
    folder = tmp_path/'run'
    first = run_workflow_task(task,cfg,folder,reader_name='ocr',api_factory=DemoAPI,counter=lambda m,t:100)
    assert first['status']=='paused'
    monkeypatch.setattr(RepairWorkflow,'finalize',original)
    def no_model(*args,**kwargs):
        raise AssertionError('Must not run model after reference grading started')
    second = run_workflow_task(task,cfg,folder,reader_name='ocr',resume=True,api_factory=no_model,counter=lambda m,t:100)
    assert second['status']=='verified'
    assert second['usage']['code_model_calls']==first['usage']['code_model_calls']


def test_html_escapes_inputs_and_delivery_excludes_private_inputs(tmp_path):
    atomic_json(tmp_path/'workflow.json',{'problem':'<script>alert(1)</script>','status':'paused','plan':[]})
    (tmp_path/'patch.diff').write_text('+<img src=x onerror=alert(1)>')
    (tmp_path/'private_gold.json').write_text('SECRET')
    render_run(tmp_path)
    page = (tmp_path/'report.html').read_text()
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page
    export_delivery(tmp_path)
    assert set(zipfile.ZipFile(tmp_path/'delivery.zip').namelist())=={'report.html','patch.diff'}


def test_batch_counts_errors_in_fixed_denominator(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_repair_workflow import report

    from screenshot_agent.harness import PROTOCOL
    atomic_json(tmp_path/'prepared.json',{'protocol':PROTOCOL,'tasks':[{'task_id':str(i)} for i in range(100)]})
    for i in range(100):
        atomic_json(tmp_path/'runs'/str(i)/'result.json',{'task_id':str(i),'method':METHOD,'protocol':PROTOCOL,
            'status':'error' if i==99 else 'verified' if i==0 else 'unresolved','resolved':i==0,'runtime_failure':i==99})
    value = report(tmp_path)
    assert value['complete'] and value['resolved']==1 and value['total']==100 and value['runtime_failures']==1


def test_batch_uses_one_service_and_skips_all_finished_attempts(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_repair_workflow as runner

    runtime = tmp_path/'runtime'
    atomic_json(runtime/'source_state.json',{'sha256':{}})
    atomic_json(runtime/'configs/scale_4gpu_resources.json',{})
    tasks = [{'task_id':str(i),'task_file':str(tmp_path/f'input/{i}/task.json')} for i in range(100)]
    atomic_json(tmp_path/'prepared.json',{'protocol':runner.PROTOCOL,'tasks':tasks})
    args = runner.parser().parse_args(['evaluate','--output',str(tmp_path)])
    calls = []
    service = {'proc':SimpleNamespace(poll=lambda:None)}
    def start(c,args,runtime,arm,pins):
        calls.append(('service',arm,args.max_num_seqs,args.service_context_tokens))
        return service,{'SCA_CODE_API_KEY':'test-only'}
    def launch(command,label,**kwargs):
        folder = Path(command[command.index('--run-dir')+1])
        row = {'task_id':folder.name,'method':METHOD,'protocol':runner.PROTOCOL,'status':'unresolved','resolved':False}
        atomic_json(folder/'result.json',row)
        atomic_json(folder/'config.json',json.loads((tmp_path/'workflow_config.json').read_text()))
        calls.append(('task',folder.name))
        return {'proc':SimpleNamespace(poll=lambda:0,returncode=0),'log':'test-only'}
    monkeypatch.setattr(runner.scale,'frozen_runtime',lambda output:runtime)
    monkeypatch.setattr(runner.scale,'verify_resource_receipts',lambda:{'fixture':True})
    monkeypatch.setattr(runner.harness,'framework_environment',lambda:{'fixture':True})
    monkeypatch.setattr(runner,'validate_prepared',lambda p:None)
    monkeypatch.setattr(runner.scale,'start_service',start)
    monkeypatch.setattr(runner.scale,'terminate',lambda proc:calls.append(('terminated',)))
    monkeypatch.setattr(runner.harness,'capacity_check',lambda *a:calls.append(('capacity',)))
    c = SimpleNamespace(update=lambda **kw:None,launch=launch)
    runner.evaluate(c,args)
    assert calls[:2]==[('service','coder',2,32768),('capacity',)]
    assert sum(x[0]=='service' for x in calls)==1
    assert sum(x[0]=='task' for x in calls)==100
    runner.evaluate(c,args)
    assert sum(x[0]=='task' for x in calls)==100


@pytest.mark.sandbox
def test_changed_source_resume_rejected_without_overwriting_paused_result(tmp_path,monkeypatch):
    task = basic_task(tmp_path)
    monkeypatch.setattr(DemoAPI,'script',[RunCancelled('pause before plan')])
    folder = tmp_path/'run'
    first = run_workflow_task(task,config(),folder,reader_name='ocr',api_factory=DemoAPI,counter=lambda m,t:100)
    assert first['status']=='paused'
    before = (folder/'result.json').read_bytes()
    monkeypatch.setattr('screenshot_agent.workflow.implementation',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='Implementation changed'):
        run_workflow_task(task,config(),folder,reader_name='ocr',resume=True)
    assert (folder/'result.json').read_bytes()==before
