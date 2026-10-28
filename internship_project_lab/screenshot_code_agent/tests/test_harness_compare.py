import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError
from openai.types.chat import ChatCompletion

from screenshot_agent.config import Task
from screenshot_agent.control import BudgetExceeded, Events, atomic_json
from screenshot_agent.harness import PROTOCOL, HarnessAPIModel, HarnessBudget, HarnessConfig, load_mini, make_mini
from screenshot_agent.harness_runner import run_harness_task
from screenshot_agent.hidden_eval import grade_hidden, prepare_hidden_task, validate_hidden_task, validate_prepared
from screenshot_agent.models import APIModel
from screenshot_agent.repository import git
from screenshot_agent.scale_data import sha256, tree_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def old_input(tmp_path):
    folder = tmp_path / 'old'
    repo = folder / 'repo'
    repo.mkdir(parents=True)
    (repo / 'miniapp.py').write_text('def add(a,b):\n    return a-b\n')
    patch = ('diff --git a/hidden_spec.py b/hidden_spec.py\nnew file mode 100644\n'
             '--- /dev/null\n+++ b/hidden_spec.py\n@@ -0,0 +1 @@\n+SECRET_REFERENCE_TEST = 928617\n')
    git(repo, 'init', '-q')
    git(repo, 'apply', '-', stdin=patch.encode())
    (repo / '.sca_eval.sh').write_text("python3 -c 'import miniapp; assert miniapp.add(2,3)==5; print(\"hidden-ok\")'\n")
    metadata = {'record':{'instance_id':'synthetic_hidden', 'test_patch':patch, 'patch':'PRIVATE_GOLD'},
                'harness_checkout':'unused'}
    atomic_json(folder / 'benchmark.json', metadata)
    task = Task(task_id='synthetic_hidden', repo=str(repo), problem='Fix addition', protected_paths=[],
                benchmark=str(folder / 'benchmark.json'), test_command='bash .sca_eval.sh')
    atomic_json(folder / 'task.json', task.model_dump())
    audit = {'status':'validated','task_sha256':sha256(folder / 'task.json'),
             'benchmark_sha256':sha256(folder / 'benchmark.json'), 'repo_sha256':tree_hash(repo)}
    atomic_json(folder / 'audit.json', audit)
    return folder


def configuration(framework='corecoder'):
    data = json.loads((ROOT / 'configs/harness-compare.json').read_text())
    data['framework'] = framework
    data['code']['model'] = 'scripted-integration-only'
    return HarnessConfig.model_validate(data)


def test_hidden_preparation_roundtrip_and_original_unchanged(old_input, tmp_path):
    old_hash = tree_hash(old_input / 'repo')
    folder = tmp_path / 'prepared'
    receipt = prepare_hidden_task(old_input / 'task.json', folder)
    assert receipt['reference_roundtrip_verified']
    assert not (folder / 'repo/hidden_spec.py').exists()
    assert not (folder / 'repo/.sca_eval.sh').exists()
    assert tree_hash(old_input / 'repo') == old_hash
    assert (old_input / 'repo/hidden_spec.py').is_file()
    validate_prepared(folder)
    assert prepare_hidden_task(old_input / 'task.json', folder) == receipt
    (folder / 'repo/miniapp.py').write_text('tampered')
    with pytest.raises(ValueError, match='repository changed'):
        validate_prepared(folder)


def test_visible_protocol_rejected(old_input):
    with pytest.raises(ValueError, match='visible-reference-test'):
        validate_hidden_task(Task.from_file(old_input / 'task.json'))


def test_hidden_metadata_cannot_live_inside_repo(old_input, tmp_path):
    folder = tmp_path / 'clean'
    prepare_hidden_task(old_input / 'task.json', folder)
    task = Task.from_file(folder / 'task.json')
    path = folder / 'repo/private.json'
    path.write_text(Path(task.benchmark).read_text())
    with pytest.raises(ValueError, match='outside'):
        validate_hidden_task(task.model_copy(update={'benchmark':str(path)}))


def test_token_limits_include_tools_and_failed_requests(tmp_path):
    config = configuration().model_copy(update={'total_prompt_tokens':15, 'total_completion_tokens':10000})
    seen = []
    budget = HarnessBudget(config, Events(tmp_path / 'events'), lambda m,t: seen.append(t) or 10)
    reserve = budget.reserve('code', config.code, [], [{'name':'tool'}])
    budget.settle(config.code, reserve, None)
    assert seen == [[{'name':'tool'}]]
    assert budget.accounted_prompt == 10
    assert budget.accounted_completion == config.code.max_output_tokens
    with pytest.raises(BudgetExceeded, match='prompt token'):
        budget.reserve('code', config.code, [], [])


def test_full_prompt_limit_and_context_configuration(tmp_path):
    config = configuration()
    budget = HarnessBudget(config, Events(tmp_path / 'events'), lambda m,t: config.max_prompt_tokens+1)
    with pytest.raises(BudgetExceeded, match='including tools'):
        budget.reserve('code', config.code, [], [])
    data = config.model_dump()
    data['service_context_tokens'] = 4096
    with pytest.raises(ValueError, match='exceed'):
        HarnessConfig.model_validate(data)


@pytest.mark.parametrize('always_fails',[False,True])
def test_transport_retry_is_bounded_and_charged_to_same_budget(tmp_path,always_fails):
    cfg=configuration()
    events=Events(tmp_path/'events')
    budget=HarnessBudget(cfg,events,lambda m,t:10)
    calls=[]
    response=ChatCompletion.model_validate({'id':'scripted','object':'chat.completion','created':0,
        'model':'scripted','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'OK'}}],
        'usage':{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}})
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==1 or always_fails:
            raise APITimeoutError(request=httpx.Request('POST','http://127.0.0.1/v1/chat/completions'))
        return response
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options=lambda **kw:client
    api=object.__new__(HarnessAPIModel)
    api.config,api.budget,api.events,api.kind,api.policy,api.client=cfg.code,budget,events,'code',cfg.agent,client
    if always_fails:
        with pytest.raises(APITimeoutError):
            api.complete([{'role':'user','content':'hello'}])
    else:
        assert api.complete([{'role':'user','content':'hello'}]).choices[0].message.content=='OK'
    assert len(calls)==2 and budget.code_calls==2
    assert budget.accounted_prompt==20
    assert budget.accounted_completion==(8192 if always_fails else 4097)


def test_mini_is_actual_pinned_upstream(tmp_path):
    cls = load_mini(tmp_path)
    assert cls.__module__ == 'minisweagent.agents.default'
    assert 'vendor/mini-swe-agent/' in sys.modules[cls.__module__].__file__


class ScriptedAPI(APIModel):
    commands = []

    def __init__(self, config, budget, events, **kwargs):
        self.config, self.budget, self.events = config, budget, events
        self.policy = kwargs['policy']
        self.script = iter(type(self).commands)
        self.seen = []

    def complete(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        text = json.dumps(messages)
        assert 'SECRET_REFERENCE_TEST' not in text
        assert '928617' not in text
        assert 'PRIVATE_GOLD' not in text
        reserve = self.budget.reserve('code', self.config, messages, tools)
        self.budget.settle(self.config, reserve, SimpleNamespace(prompt_tokens=10, completion_tokens=10))
        item = next(self.script)
        message = {'role':'assistant','content':item} if isinstance(item,str) else {
            'role':'assistant','content':'Execute the next repair action.',
            'tool_calls':[{'id':f'call{len(self.seen)}','type':'function',
                           'function':{'name':item[0],'arguments':json.dumps(item[1])}}]}
        return ChatCompletion.model_validate({'id':'scripted', 'object':'chat.completion','created':0,
            'model':'scripted-integration-only','choices':[{'index':0,'finish_reason':'stop','message':message}]})


@pytest.mark.sandbox
@pytest.mark.parametrize('framework', ['corecoder','mini'])
def test_both_frameworks_repair_without_reference_feedback(old_input, tmp_path, monkeypatch, framework):
    folder = tmp_path / 'clean'
    prepare_hidden_task(old_input / 'task.json', folder)
    task = Task.from_file(folder / 'task.json')
    proof = 'test ! -e hidden_spec.py && test ! -e .sca_eval.sh && test ! -e /grader/eval.sh && echo clean-input'
    if framework == 'mini':
        ScriptedAPI.commands = [('bash',{'command':proof}),
            ('bash',{'command':"python3 -c \"from pathlib import Path; p=Path('miniapp.py'); p.write_text(p.read_text().replace('a-b','a+b'))\""}),
            ('bash',{'command':'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'})]
    else:
        ScriptedAPI.commands = [('bash',{'command':proof}),
            ('edit_file',{'file_path':'miniapp.py','old_string':'return a-b','new_string':'return a+b'}), 'Done']
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade',
        lambda metadata,log: {'resolved':'hidden-ok' in log.read_text(), 'parsed':True})
    directory = tmp_path / ('run_' + framework)
    def grade_after_termination(*args, **kwargs):
        assert (directory / 'conversation.json').exists()
        assert not (directory / 'workspace/hidden_spec.py').exists()
        return grade_hidden(*args, **kwargs)
    result = run_harness_task(task, configuration(framework), directory,
                             counter=lambda m,t:100, api_factory=ScriptedAPI, grader=grade_after_termination)
    assert result['status'] == 'verified', result
    assert result['changed_files'] == ['miniapp.py']
    assert not (folder / 'repo/hidden_spec.py').exists()
    assert not (directory / 'workspace/hidden_spec.py').exists()
    assert (directory / 'verification/hidden_spec.py').exists()
    assert 'return a-b' in (old_input / 'repo/miniapp.py').read_text()
    assert result['protocol'] == PROTOCOL


@pytest.mark.sandbox
def test_mini_plain_text_does_not_submit(tmp_path, sandbox):
    ScriptedAPI.commands = ['I am done', ('bash',{'command':'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'})]
    cfg = configuration('mini')
    budget = HarnessBudget(cfg, sandbox.events, lambda m,t:10)
    api = ScriptedAPI(cfg.code,budget,sandbox.events,policy=cfg.agent)
    agent = make_mini(cfg,budget,sandbox.events,sandbox,tmp_path,api=api)
    result = agent.run('Inspect this repository')
    assert result['exit_status'] == 'Submitted'
    assert budget.code_calls == 2
    assert any(m.get('extra',{}).get('interrupt_type') == 'FormatError' for m in agent.messages)


@pytest.mark.sandbox
def test_search_truncation_is_visible_and_git_is_excluded(sandbox):
    for i in range(205):
        (sandbox.repo / f'item{i:03d}.py').write_text('needle\n')
    (sandbox.repo / '.git').mkdir()
    (sandbox.repo / '.git/private').write_text('secret-needle')
    result = json.loads(json.loads(sandbox.filesystem('grep',{'pattern':'needle'}))['output'])
    assert result['returned_matches'] == 200
    assert result['truncated'] is True
    assert '.git/' not in result['output']


def test_report_preserves_errors_in_fixed_denominator(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_harness_compare import report, storage
    tasks = [{'task_id':str(i)} for i in range(100)]
    atomic_json(tmp_path/'prepared.json',{'tasks':tasks,'protocol':PROTOCOL})
    for arm in ('corecoder','mini'):
        for i in range(100):
            atomic_json(tmp_path/arm/'runs'/str(i)/'result.json',
                {'task_id':str(i),'status':'error' if i==99 else 'unresolved',
                 'resolved':i==0,'runtime_failure':i==99,'protocol':PROTOCOL,'framework':arm})
    data=report(tmp_path)
    assert data['complete']
    assert data['arms']['mini']['total']==100
    assert data['arms']['mini']['runtime_failures']==1
    with pytest.raises(ValueError,match='/data/zjr'):
        storage('/tmp/not-approved-data',create=False)


def test_server_resources_are_configurable_without_changing_legacy_defaults():
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_scale_4gpu import server_command
    pin={'tool_call_parser':'qwen3_coder'}
    legacy=server_command(ROOT,'coder','2,5,6,7',18846,pin)
    new=server_command(ROOT,'coder','2,5,6,7',18846,pin,context_tokens=32768,max_num_seqs=4)
    assert legacy[legacy.index('--max-model-len')+1]=='16384'
    assert legacy[legacy.index('--max-num-seqs')+1]=='2'
    assert new[new.index('--max-model-len')+1]=='32768'
    assert new[new.index('--max-num-seqs')+1]=='4'


def test_capacity_probe_precedes_tasks_and_never_saves_credentials(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_harness_compare import capacity_check
    updates=[]
    c=SimpleNamespace(update=lambda **kw:updates.append(kw))
    args=SimpleNamespace(workers=2,port=18846,output=tmp_path)
    calls=[]
    def chat(payload):
        calls.append(payload)
        return {'choices':[{'message':{'content':'OK'},'finish_reason':'stop'}]}
    value=capacity_check(c,args,configuration(),'secret-not-for-disk',chat=chat,counter=lambda m,t:24064)
    assert len(calls)==2 and value['concurrency']==2
    assert updates[-1]['probe_completed']==2
    assert 'secret-not-for-disk' not in (tmp_path/'capacity_check.json').read_text()
