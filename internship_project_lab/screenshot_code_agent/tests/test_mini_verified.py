"""Regression coverage for actual dev5 failures; no model or gold patch execution."""
import json
import os
from pathlib import Path

import pytest
from test_harness_compare import ScriptedAPI, configuration
from test_harness_compare import old_input as inherited_old_input
from test_mini_session import ROOT, SELECTED

from screenshot_agent.config import Task
from screenshot_agent.control import Events
from screenshot_agent.harness import HarnessBudget, MiniEnvironment, load_mini
from screenshot_agent.harness_runner import install_development_case, run_harness_task
from screenshot_agent.hidden_eval import grade_hidden, prepare_hidden_task
from screenshot_agent.mini_actions import execute_action
from screenshot_agent.mini_verified import VerifiedDelivery
from screenshot_agent.public_display import PublicDisplay
from screenshot_agent.repository import git, prepare_repository
from screenshot_agent.runtime import create_sandbox
from screenshot_agent.scale_data import tree_hash

old_input = inherited_old_input


@pytest.fixture(autouse=True)
def synthetic_repo_does_not_load_unrelated_plotting_plugin(monkeypatch):
    # The PyVista image auto-loads its plugin, which imports the source repository.
    # Our tiny synthetic repository is not PyVista; real-repository checks keep it enabled.
    original = create_sandbox
    def factory(repo, *args, **kwargs):
        sb = original(repo, *args, **kwargs)
        run = sb.run
        def execute(command, **options):
            if (Path(repo)/'test_public.py').exists() and isinstance(command, list) and 'pytest' in command:
                command = [*command, '-p', 'no:pyvista']
            return run(command, **options)
        sb.run = execute
        return sb
    monkeypatch.setattr('screenshot_agent.mini_verified.create_sandbox', factory)


@pytest.fixture(params=[None, *SELECTED], ids=lambda r: r['task_id'] if r else 'host-venv')
def runtime(request):
    return (Task.from_file(Path(request.param['task_file'])).runtime if request.param else
            Task.from_file(Path(json.loads(Path('/data/zjr/screenshot_code_agent/mini_session_dev5_20261029/prepared.json').read_text())['tasks'][0]['task_file'])).runtime)


@pytest.fixture
def checked_session(tmp_path, runtime):
    load_mini(tmp_path)
    source = tmp_path/'source'
    source.mkdir()
    (source/'miniapp.py').write_text('def add(a,b):\n    return a-b\n')
    (source/'test_public.py').write_text('from miniapp import add\ndef test_identity():\n    assert add(2,0)==2\n')
    directory = tmp_path/'run'
    baseline, workspace = prepare_repository(source, directory)
    events = Events(directory/'events.jsonl')
    for root in (baseline, workspace):
        (root/'.sca_scratch').mkdir()
        (root/'.sca_scratch/check.py').write_text('from miniapp import add\nassert add(2,3)==5\n')
    protected = ['test_public.py', '.sca_scratch/check.py']
    sb = create_sandbox(workspace, events, protected, 10, baseline, runtime)
    sb.enable_agent_session()
    cfg = configuration('mini').model_copy(update={'mini_session': True, 'mini_verified': True})
    budget = HarnessBudget(cfg, events, lambda m,t:100)
    delivery = VerifiedDelivery(baseline, workspace, directory, protected, {}, sb, budget,
                               {'path': '.sca_scratch/check.py', 'public_modules': ['test_public.py']}, runtime=runtime)
    yield sb, budget, delivery
    sb.cancel()


@pytest.mark.sandbox
def test_invalid_arguments_are_feedback_and_next_tool_still_runs(checked_session):
    sb, budget, delivery = checked_session
    env = MiniEnvironment(sb, budget, sb.events, delivery)
    for command in ('printf "bad\0argument"', None, ['echo', 'bad-type']):
        out = env.execute({'command': command})
        assert out['returncode'] != 0 and out['input_error']
    assert env.execute({'command': 'echo recovered'})['output'].strip() == 'recovered'
    assert env.recoverable_errors == 3 and budget.tool_calls == 4


@pytest.mark.sandbox
def test_exact_edit_is_unique_syntax_checked_and_does_not_destroy_source(checked_session):
    sb, budget, _ = checked_session
    original = (sb.repo/'miniapp.py').read_text()
    action = {'tool': 'edit_file', 'file_path': 'miniapp.py', 'old_string': 'return a-b'}
    assert execute_action({**action, 'new_string': 'return ('}, sb, budget, None)['returncode'] != 0
    assert (sb.repo/'miniapp.py').read_text() == original
    assert execute_action({**action, 'old_string': 'missing', 'new_string': 'pass'}, sb, budget, None)['returncode'] != 0
    assert (sb.repo/'miniapp.py').read_text() == original
    assert execute_action({**action, 'new_string': 'return a+b'}, sb, budget, None)['returncode'] == 0
    assert 'return a+b' in (sb.repo/'miniapp.py').read_text()


@pytest.mark.sandbox
def test_public_regression_failure_blocks_delivery_even_when_behavior_passes(checked_session):
    sb, _, delivery = checked_session
    delivery.initialize()
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return 5\n')
    checked = delivery()
    assert not checked['passed']
    report = json.loads((delivery.directory/'delivery_check.json').read_text())
    assert report['visible_development_case']['returncode'] == 0
    assert report['comparison']['status'] == 'failed'
    assert delivery.best is None


@pytest.mark.sandbox
def test_checkpoint_preserved_after_bad_edit_and_cached_check_bound_to_patch(checked_session):
    sb, _, delivery = checked_session
    delivery.initialize()
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    assert delivery()['passed']
    attempts = delivery.attempts
    assert delivery()['passed'] and delivery.attempts == attempts
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return 5\n')
    assert not delivery()['passed']
    selected = delivery.select_final(delivery.directory/'final.diff')
    assert selected['candidate_selection']['used_earlier_candidate']
    assert 'return a+b' in (delivery.directory/'final.diff').read_text()
    assert 'return 5' in (delivery.directory/'last_workspace_candidate.diff').read_text()


@pytest.mark.sandbox
def test_real_mini_tools_recover_and_keep_verified_candidate_at_call_limit(old_input, tmp_path, runtime, monkeypatch):
    repo = old_input/'repo'
    (repo/'test_public.py').write_text('from miniapp import add\ndef test_identity():\n    assert add(2,0)==2\n')
    audit = json.loads((old_input/'audit.json').read_text())
    audit['repo_sha256'] = tree_hash(repo)
    (old_input/'audit.json').write_text(json.dumps(audit))
    prepared = tmp_path/'prepared'
    prepare_hidden_task(old_input/'task.json', prepared, runtime=runtime)
    task = Task.from_file(prepared/'task.json').model_copy(update={'protected_paths': ['test_public.py']})
    cfg = configuration('mini').model_copy(update={'mini_session': True, 'mini_verified': True})
    cfg.limits.max_steps = cfg.limits.max_model_calls = 4
    ScriptedAPI.commands = [
        ('bash', {'command': 'printf "bad\0argument"'}),
        ('edit_file', {'file_path': 'miniapp.py', 'old_string': 'return a-b', 'new_string': 'return a+b'}),
        ('check_patch', {}),
        ('bash', {'command': "printf 'def add(a,b):\n    return 5\n' > miniapp.py"}),
    ]
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade',
        lambda metadata,log: {'resolved': 'hidden-ok' in log.read_text(), 'parsed': True})
    result = run_harness_task(task, cfg, tmp_path/'actual-mini', counter=lambda m,t:100,
                             api_factory=ScriptedAPI, grader=grade_hidden)
    assert result['resolved'], result
    assert result['exit_reason'] == 'budget_exceeded'
    assert result['recoverable_tool_errors'] == 1
    assert result['candidate_selection']['used_earlier_candidate']
    assert result['usage']['code_model_calls'] == 4


@pytest.mark.sandbox
@pytest.mark.parametrize('row', SELECTED, ids=[row['task_id'] for row in SELECTED])
def test_previous_bad_patch_rejected_in_actual_environment(row, tmp_path):
    task = Task.from_file(Path(row['task_file']))
    tree = {'max_files': task.runtime.max_repo_files, 'max_bytes': task.runtime.max_repo_bytes,
            'allowed_symlinks': task.runtime.allowed_symlinks}
    baseline, workspace = prepare_repository(Path(task.repo), tmp_path/'actual', **tree)
    repair = os.environ.get('SCA_MINI_REPAIR_PREFLIGHT') == '1'
    case = install_development_case(task, baseline, workspace, verified=True, repair=repair)
    protected = [*task.protected_paths, case['path']]
    events = Events(tmp_path/'events.jsonl')
    sb = create_sandbox(workspace, events, protected, 90, baseline, task.runtime)
    sb.enable_agent_session()
    cfg = configuration('mini').model_copy(update={'mini_session': True, 'mini_verified': True})
    cfg.mini_repair_control = repair
    budget = HarnessBudget(cfg, events, lambda m,t:100)
    display = None
    if task.task_id.startswith('pyvista'):
        conf = json.loads((ROOT/'configs/mini-verified-dev5.json').read_text())
        display = PublicDisplay(conf['xvfb_executable'], tmp_path)
        sb.public_display = display
    delivery = VerifiedDelivery(baseline, workspace, tmp_path/'actual', protected, tree, sb, budget, case,
                                runtime=task.runtime, problem=task.problem)
    try:
        delivery.initialize()
        run_name = 'mini_verified_dev5_20261029' if repair else 'mini_session_dev5_20261029'
        previous = Path('/data/zjr/screenshot_code_agent')/run_name/'mini/runs'/task.task_id/'patch.diff'
        git(workspace, 'apply', '-', stdin=previous.read_bytes())
        assert delivery()['passed'] == (repair and task.task_id == 'pydicom__pydicom-793'), task.task_id
        # Positive controls use previously successful model patches, not reference patches.
        if task.task_id in {'pylint-dev__astroid-984', 'pydicom__pydicom-793'}:
            git(workspace, 'apply', '-R', '-', stdin=previous.read_bytes())
            good = Path('/data/zjr/screenshot_code_agent/v4_dev10_20261029/runs')/task.task_id/'patch.diff'
            git(workspace, 'apply', '-', stdin=good.read_bytes())
            assert delivery()['passed'], (task.task_id, json.loads((delivery.directory/'delivery_check.json').read_text()))
    finally:
        sb.cancel()
        if display:
            display.close()


def test_config_rejects_verified_mode_without_session():
    from screenshot_agent.harness import HarnessConfig
    data = configuration('mini').model_dump()
    data['mini_verified'] = True
    with pytest.raises(ValueError, match='requires mini_session'):
        HarnessConfig.model_validate(data)


def test_no_reference_access_in_verified_components():
    # Keep verification implementation independent of benchmark and hidden-grader APIs.
    import screenshot_agent.mini_cases as cases
    import screenshot_agent.mini_verified as verified
    for module in (cases, verified):
        source = Path(module.__file__).read_text()
        assert 'grade_hidden(' not in source and '.benchmark' not in source
