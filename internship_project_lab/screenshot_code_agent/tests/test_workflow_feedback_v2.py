import copy
import json
import sys
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from test_repair_workflow import basic_task, config

from screenshot_agent.config import Task
from screenshot_agent.context import RepairContext, compact_request
from screenshot_agent.control import ContextWindowExceeded, Events
from screenshot_agent.harness import HarnessAPIModel, HarnessBudget
from screenshot_agent.workflow import RepairWorkflow, run_workflow_task
from screenshot_agent.workflow_demo import PLAN, DemoAPI, patch
from screenshot_agent.workflow_feedback import compare_public, junit_statuses

ROOT = Path(__file__).resolve().parents[1]


def chars(messages, tools):
    return len(json.dumps(messages)) + len(json.dumps(tools))


def test_exact_context_preserves_issue_latest_request_and_tool_pairs(tmp_path):
    history = [{'role':'user','content':'Original immutable issue'},
               {'role':'assistant','tool_calls':[{'id':'a','type':'function',
                 'function':{'name':'read','arguments':'{}'}},{'id':'b','type':'function',
                 'function':{'name':'read','arguments':'{}'}}]},
               {'role':'tool','tool_call_id':'a','content':'source ' * 1500},
               {'role':'tool','tool_call_id':'b','content':'errors ' * 1500},
               {'role':'user','content':'Current phase: repair; preserve this request'}]
    system = [{'role':'system','content':'System instructions ' * 20}]
    tools = [{'name':'read','description':'schema ' * 30}]
    original = copy.deepcopy(history)
    context = RepairContext(3000,history[0]['content'],Events(tmp_path/'events'))
    context.bind_request(chars,3000,lambda:system,lambda:tools)
    assert context.maybe_compress(history)
    assert chars(system+history,tools) <= 3000
    assert history[0] == original[0] and history[-1] == original[-1]
    ids = {call['id'] for row in history for call in row.get('tool_calls',[])}
    assert {row['tool_call_id'] for row in history if row['role']=='tool'} == ids
    assert system[0]['content'] == 'System instructions ' * 20


def test_uncompressible_issue_is_explicit_and_does_not_mutate_input(tmp_path):
    messages = [{'role':'system','content':'instructions'}, {'role':'user','content':'x' * 3000}]
    before = copy.deepcopy(messages)
    with pytest.raises(ContextWindowExceeded,match='cannot fit'):
        compact_request(messages,[],chars,1000,Events(tmp_path/'events'))
    assert messages == before


def test_full_request_is_compacted_before_any_api_charge(tmp_path):
    cfg=config()
    events=Events(tmp_path/'events')
    budget=HarnessBudget(cfg,events,chars)
    sent=[]
    def respond(request):
        payload=json.loads(request.content)
        sent.append(payload)
        return httpx.Response(200,json={'id':'test','object':'chat.completion','created':0,'model':'test',
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'done'}}],
            'usage':{'prompt_tokens':2000,'completion_tokens':1,'total_tokens':2001}})
    client=OpenAI(api_key='test-only',base_url='http://local.invalid/v1',max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    api=object.__new__(HarnessAPIModel)
    api.config,api.budget,api.events,api.kind,api.policy,api.client=cfg.code,budget,events,'code',cfg.agent,client
    messages=[{'role':'system','content':'preserve system'}, {'role':'user','content':'preserve issue'},
              {'role':'assistant','tool_calls':[{'id':'read1','type':'function',
                  'function':{'name':'read','arguments':'{}'}}]},
              {'role':'tool','tool_call_id':'read1','content':'observed source '*4000}]
    tools=[{'type':'function','function':{'name':'read','description':'schema '*100,
                                        'parameters':{'type':'object','properties':{}}}}]
    api.complete(messages,tools)
    assert len(sent)==1 and budget.code_calls==1 and budget.summary_calls==0
    assert chars(sent[0]['messages'],sent[0]['tools']) <= cfg.max_prompt_tokens
    assert sent[0]['messages'][:2]==messages[:2] and sent[0]['tools']==tools
    assert len(messages[-1]['content']) > cfg.max_prompt_tokens  # API preparation is non-destructive.
    with pytest.raises(ContextWindowExceeded):
        api.complete([{'role':'user','content':'required ' * 8000}],tools)
    assert len(sent)==1 and budget.code_calls==1
    client.close()


def result(cases, code=1, **extra):
    return {'cases':cases,'returncode':code,'timed_out':False,'structured':True,**extra}


def test_existing_failures_are_not_new_regressions():
    baseline = result({'stable':'passed','offline_network':'failed'})
    assert compare_public(baseline,result(dict(baseline['cases'])))['status']=='passed'
    new = result({'stable':'failed','offline_network':'failed'})
    compared = compare_public(baseline,new)
    assert compared['status']=='failed' and compared['lost_passing_tests']==['stable']


@pytest.mark.parametrize('cases', [{}, {'stable':'skipped'}, {'another':'passed'}])
def test_missing_or_skipped_passing_cases_are_not_a_pass(cases):
    assert compare_public(result({'stable':'passed'},0),result(cases,0))['status']!='passed'


def test_timeout_and_patch_replay_failure_are_inconclusive():
    base = result({'stable':'passed'},0)
    assert compare_public(base,result({'stable':'passed'},0,timed_out=True))['status']=='inconclusive'
    assert compare_public(base,result({'stable':'passed'},0,patch_replay={'applied':False}))['status']=='inconclusive'


def test_junit_ids_duplicates_and_parse_errors(tmp_path):
    path = tmp_path/'report.xml'
    path.write_text('<testsuites><testsuite><testcase classname="x" name="ok"/>'
                    '<testcase classname="x" name="bad"><failure/></testcase></testsuite></testsuites>')
    assert junit_statuses(path)=={'x::ok':'passed','x::bad':'failed'}
    path.write_text('<testsuite><testcase name="same"/><testcase name="same"/></testsuite>')
    assert junit_statuses(path) is None
    path.write_text('not xml')
    assert junit_statuses(path) is None


@pytest.fixture
def flow(tmp_path):
    repo=tmp_path/'input'
    (repo/'tests').mkdir(parents=True)
    (repo/'maths.py').write_text('def add(a,b):\n    return a-b\ndef stable():\n    return 7\n')
    (repo/'tests/test_public.py').write_text('from maths import stable\n'
        'def test_stable():\n    assert stable()==7\n'
        'def test_preexisting():\n    assert False, "pre-existing environment constraint"\n')
    task=Task(task_id='visible_feedback',repo=str(repo),problem='Fix add to return a+b',
              test_command="python3 -c 'from maths import stable; assert stable()==7'",protected_paths=['tests'])
    folder=tmp_path/'run'
    folder.mkdir()
    f=RepairWorkflow(task,config(),folder,counter=lambda m,t:100,model_inference=False)
    f.setup()
    yield f
    f.sandbox.cancel()


def register(flow):
    path=flow.workspace/'repro.py'
    path.write_text('from maths import add\nassert add(2,3)==5, "addition returns the wrong sum"\n')
    return flow.feedback.register_reproduction('repro.py','addition returns the wrong sum')


@pytest.mark.sandbox
def test_real_checks_use_frozen_reproducer_and_cached_baseline(flow):
    baseline=flow.feedback.baseline()
    assert baseline['returncode']==0
    assert register(flow)['baseline_reproduced']
    # A model-generated root debug file must not contaminate existing regression tests.
    (flow.workspace/'test_debug.py').write_text('import nonexistent_debug_dependency\n')
    file=flow.workspace/'maths.py'
    file.write_text(file.read_text().replace('return a-b','return a+b'))
    check=flow.feedback.check('fixed')
    assert check['returncode']==0
    assert check['regression']['status']=='passed'
    assert check['reproduction']['status']=='verified'
    assert check['passed'] and not check['retry_needed']
    assert flow.feedback.baseline() == baseline
    assert len(list((flow.directory/'public_checks').glob('baseline_*')))==1


@pytest.mark.sandbox
def test_prepared_oci_image_runs_real_pytest_baseline(tmp_path):
    # Optional local integration using an existing read-only official runtime.
    # No image download, model requests, or access to reference test metadata.
    prepared=Path('/data/zjr/screenshot_code_agent/harness_compare_20261027/data/tasks/'
                  'marshmallow-code__marshmallow-2123/task.json')
    if not prepared.exists():
        pytest.skip('Prepared local OCI runtime is not available')
    runtime=Task.from_file(prepared).runtime.model_copy(update={'allowed_symlinks':{}})
    repo=tmp_path/'input'
    (repo/'tests').mkdir(parents=True)
    (repo/'maths.py').write_text('def add(a,b):\n    return a-b\n')
    (repo/'tests/test_public.py').write_text('def test_stable():\n    assert 2+3==5\n'
                                          'def test_existing_failure():\n    assert False, "existing"\n')
    task=Task(task_id='oci_pytest_smoke',repo=str(repo),problem='Fix addition',
              runtime=runtime,test_command='python -m pytest',protected_paths=['tests'])
    folder=tmp_path/'run'
    folder.mkdir()
    f=RepairWorkflow(task,config(),folder,counter=lambda m,t:100,model_inference=False)
    try:
        f.setup()
        baseline=f.feedback.baseline()
        assert baseline['returncode']==1,baseline['output']
        assert sorted(baseline['cases'].values())==['failed','passed']
        assert register(f)['baseline_reproduced']
        (f.workspace/'maths.py').write_text('def add(a,b):\n    return a+b\n')
        (f.workspace/'test_debug.py').write_text('import nonexistent_debug_dependency\n')
        checked=f.feedback.check('real pytest')
        assert checked['returncode']==1 and checked['regression']['status']=='passed'
        assert checked['reproduction']['status']=='verified' and checked['passed']
        assert not checked['retry_needed']
    finally:
        if f.sandbox:
            f.sandbox.cancel()


@pytest.mark.sandbox
def test_public_pytest_scope_comes_from_original_test_directories(flow):
    flow.task=flow.task.model_copy(update={'test_command':'python -m pytest'})
    (flow.workspace/'test_debug.py').write_text('import nonexistent_debug_dependency\n')
    command,structured=flow.feedback.public_command()
    assert structured and command.startswith('python -m pytest tests ')
    assert 'test_debug' not in command


@pytest.mark.sandbox
def test_changing_reproducer_to_pass_cannot_fake_a_fix(flow):
    assert register(flow)['baseline_reproduced']
    (flow.workspace/'repro.py').write_text('print("success")\n')
    check=flow.feedback.check('unchanged source')
    assert check['reproduction']['status']=='failed'
    assert not check['passed'] and check['retry_needed']
    assert check['non_reproduction_changed_files']==[]


@pytest.mark.sandbox
def test_changed_expected_failure_invalidates_cached_validation(flow):
    assert register(flow)['baseline_reproduced']
    (flow.workspace/'maths.py').write_text('def add(a,b):\n    return a+b\ndef stable():\n    return 7\n')
    assert flow.feedback.check('fixed')['passed']
    invalid=flow.feedback.register_reproduction('repro.py','different assertion, not observed')
    assert not invalid['baseline_reproduced']
    checked=flow.feedback.check('new expectation')
    assert not checked['passed'] and checked['reproduction']['status']=='unverified'


@pytest.mark.sandbox
def test_import_failure_is_not_an_issue_reproduction(flow):
    (flow.workspace/'repro.py').write_text('import nonexistent_module\n'
        'assert False, "target assertion"\n')
    record=flow.feedback.register_reproduction('repro.py','target assertion')
    assert not record['baseline_reproduced']
    assert 'ModuleNotFoundError' in record['output']


@pytest.mark.sandbox
def test_invalid_patch_replay_does_not_execute_candidate_tests(flow,monkeypatch):
    flow.feedback.baseline()
    (flow.workspace/'maths.py').write_text('changed source\n')
    monkeypatch.setattr('screenshot_agent.workflow_feedback.replay_patch',
                        lambda *a,**kw:{'applied':False,'error':'invalid patch'})
    def forbidden(*a,**kw):
        raise AssertionError('Must not run tests on an unapplied patch')
    monkeypatch.setattr('screenshot_agent.workflow_feedback.create_sandbox',forbidden)
    check=flow.feedback.check('invalid')
    assert not check['passed'] and check['regression']['reason']=='patch_replay_failed'


@pytest.mark.sandbox
def test_phase_allowance_does_not_mark_reproduction_complete(tmp_path,monkeypatch):
    cfg=config()
    cfg=cfg.model_copy(update={'workflow':cfg.workflow.model_copy(
        update={'locate_calls':1,'reproduce_calls':1,'max_retries':0})})
    monkeypatch.setattr(DemoAPI,'script',[PLAN,
        ('read_file',{'file_path':'maths.py'}),('read_file',{'file_path':'maths.py'}),
        ('apply_patch',{'file_path':'maths.py','patch':patch('return a-b','return a+b')}),'done'])
    folder=tmp_path/'run'
    result=run_workflow_task(basic_task(tmp_path),cfg,folder,reader_name='ocr',
                             api_factory=DemoAPI,counter=lambda m,t:100,model_inference=False)
    assert result['resolved'],result
    state=json.loads((folder/'workflow.json').read_text())
    row=next(r for r in state['plan'] if r['phase']=='reproduce')
    assert row['exit_reason']=='phase_budget' and row['status']=='incomplete'
    assert 'reproduce' not in state['completed_phases'] and 'reproduce' in state['attempted_phases']


def test_status_accepts_frozen_v1_but_rejects_mixed_results(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_repair_workflow import result_rows

    from screenshot_agent.control import atomic_json
    from screenshot_agent.harness import PROTOCOL
    tasks=[{'task_id':'old'},*[{'task_id':str(i)} for i in range(99)]]
    atomic_json(tmp_path/'prepared.json',{'protocol':PROTOCOL,'tasks':tasks})
    atomic_json(tmp_path/'workflow_plan.json',{'method':'corecoder_retrieval_workflow_v1'})
    record={'method':'corecoder_retrieval_workflow_v1','protocol':PROTOCOL,'task_id':'old',
            'status':'unresolved','resolved':False}
    atomic_json(tmp_path/'runs/old/result.json',record)
    assert result_rows(tmp_path)[0]['method']=='corecoder_retrieval_workflow_v1'
    record['method']='corecoder_retrieval_workflow_v2'
    atomic_json(tmp_path/'runs/old/result.json',record)
    with pytest.raises(ValueError,match='manifest'):
        result_rows(tmp_path)
