"""Session contracts run against the actual selected OCI images before GPU startup."""
import json
import os
import sys
from pathlib import Path

import pytest
from test_harness_compare import ScriptedAPI, configuration
from test_harness_compare import old_input as inherited_old_input

from screenshot_agent.config import RuntimeConfig, Task
from screenshot_agent.control import Events
from screenshot_agent.harness import HarnessBudget
from screenshot_agent.harness_runner import run_harness_task
from screenshot_agent.hidden_eval import grade_hidden, prepare_hidden_task
from screenshot_agent.mini_delivery import MiniDelivery
from screenshot_agent.public_display import PublicDisplay
from screenshot_agent.repository import export_session_patch, git, prepare_repository, replay_patch
from screenshot_agent.runtime import create_sandbox

old_input = inherited_old_input
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = os.environ.get('SCA_MINI_PREFLIGHT_MANIFEST')
SELECTED = json.loads(Path(MANIFEST).read_text())['tasks'] if MANIFEST else []


@pytest.fixture(params=[None, *SELECTED], ids=lambda value: value['task_id'] if value else 'host-bubblewrap')
def runtime(request):
    return Task.from_file(Path(request.param['task_file'])).runtime if request.param else RuntimeConfig()


@pytest.fixture
def session(tmp_path, runtime):
    source = tmp_path / 'source'
    (source / 'tests').mkdir(parents=True)
    (source / 'miniapp.py').write_text('def add(a,b):\n    return a-b\n')
    (source / 'tests/fixed.py').write_text('immutable = True\n')
    directory = tmp_path / 'run'
    baseline, workspace = prepare_repository(source, directory)
    events = Events(directory / 'events.jsonl')
    sandbox = create_sandbox(workspace, events, ['tests'], 10, baseline, runtime)
    sandbox.enable_agent_session()
    cfg = configuration('mini').model_copy(update={'mini_session': True})
    budget = HarnessBudget(cfg, events, lambda m,t:100)
    delivery = MiniDelivery(baseline, workspace, directory, ['tests'], {}, sandbox, budget)
    yield baseline, workspace, directory, sandbox, delivery
    sandbox.cancel()


@pytest.mark.sandbox
def test_files_survive_calls_but_shell_state_resets(session):
    baseline, workspace, directory, sandbox, delivery = session
    first = sandbox.run("printf durable > /tmp/persist.txt && printf homedata > \"$HOME/check.txt\" && cd /tmp")
    assert first['returncode'] == 0, first
    second = sandbox.run('pwd && cat /tmp/persist.txt && cat "$HOME/check.txt"')
    assert second['returncode'] == 0 and 'durablehomedata' in second['output'], second
    assert second['output'].startswith(sandbox.description()['repository_mount'] + '\n')
    assert (sandbox.session_tmp / 'persist.txt').read_text() == 'durable'


@pytest.mark.sandbox
def test_git_rollback_and_reference_isolation(session):
    baseline, workspace, directory, sandbox, delivery = session
    original = (baseline / 'miniapp.py').read_text()
    result = sandbox.run("printf 'broken (' > miniapp.py && git checkout -- miniapp.py && python3 -c 'import miniapp'")
    assert result['returncode'] == 0, result
    assert (workspace / 'miniapp.py').read_text() == original
    blocked = sandbox.run('printf changed > tests/fixed.py')
    assert blocked['returncode'] != 0 and 'Read-only' in blocked['output'], blocked
    assert (baseline / 'tests/fixed.py').read_text() == 'immutable = True\n'
    hidden = sandbox.run('test ! -e /grader/eval.sh && test -z "$SCA_CODE_API_KEY" && echo isolated')
    assert hidden['returncode'] == 0 and 'isolated' in hidden['output'], hidden


@pytest.mark.sandbox
def test_export_ignores_agent_commits_and_preserves_new_source(session):
    baseline, workspace, directory, sandbox, delivery = session
    original_head = git(baseline, 'rev-parse', 'HEAD').stdout
    (workspace / 'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    (workspace / 'helper.py').write_text('value = 3\n')
    (workspace / 'miniapp.py.backup').write_text((baseline / 'miniapp.py').read_text())
    result = sandbox.run('git add miniapp.py helper.py && git -c user.name=Probe -c user.email=probe@example.invalid commit -qm fix && mv .git /tmp/agent_git')
    assert result['returncode'] == 0, result
    exported = export_session_patch(baseline, workspace, directory/'patch.diff', ['tests'])
    assert exported['changed_files'] == ['helper.py', 'miniapp.py']
    assert exported['excluded_new_backup_files'] == ['miniapp.py.backup']
    assert git(baseline, 'rev-parse', 'HEAD').stdout == original_head
    replay = directory / 'verify'
    assert replay_patch(baseline, replay, directory/'patch.diff')['applied']
    assert (replay/'helper.py').read_text() == 'value = 3\n'
    assert 'return a+b' in (replay/'miniapp.py').read_text()
    assert not (replay/'miniapp.py.backup').exists()


@pytest.mark.sandbox
def test_submission_rejects_syntax_empty_comment_and_debug_only(session):
    baseline, workspace, directory, sandbox, delivery = session
    assert not delivery()['passed']
    (workspace/'miniapp.py').write_text('def invalid(:\n')
    assert not delivery()['passed']
    (workspace/'miniapp.py').write_text((baseline/'miniapp.py').read_text() + '# fixed!\n')
    assert not delivery()['passed']
    (workspace/'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    (workspace/'test_fix.py').write_text('print("fake success")\n')
    assert not delivery()['passed']
    (workspace/'.sca_scratch').mkdir()
    (workspace/'test_fix.py').rename(workspace/'.sca_scratch/test_fix.py')
    assert delivery()['passed']


@pytest.mark.sandbox
def test_timeout_returns_feedback_and_session_remains_usable(session):
    _, _, _, sandbox, _ = session
    result = sandbox.run('sleep 30 & wait', timeout=0.15)
    assert result['timed_out'] and result['returncode'] != 0
    assert sandbox.run('echo recovered')['output'].strip() == 'recovered'


@pytest.mark.sandbox
def test_visible_assertion_and_package_import_block_false_success(session):
    baseline, workspace, directory, sandbox, delivery = session
    (workspace/'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    (workspace/'.sca_scratch').mkdir()
    (baseline/'.sca_scratch').mkdir()
    for root in (workspace, baseline):
        (root/'.sca_scratch/check.py').write_text('from miniapp import add\nassert add(2,3)==6\n')
    sandbox.protected.append('.sca_scratch/check.py')
    delivery.case = {'path': '.sca_scratch/check.py'}
    assert not delivery()['passed']
    assert sandbox.run('printf weakened > .sca_scratch/check.py')['returncode'] != 0
    (workspace/'miniapp.py').write_text('def add(a,b):\n    return a+b+1\n')
    assert delivery()['passed']
    (workspace/'broken_package').mkdir()
    (workspace/'broken_package/__init__.py').write_text('import absent_dependency_for_probe\n')
    assert not delivery()['passed']


@pytest.mark.sandbox
def test_actual_mini_rejects_bad_submit_then_recovers_and_grades(old_input, tmp_path, runtime, monkeypatch):
    prepared = tmp_path / 'prepared'
    prepare_hidden_task(old_input/'task.json', prepared, runtime=runtime)
    task = Task.from_file(prepared/'task.json')
    cfg = configuration('mini').model_copy(update={'mini_session': True, 'final_test_timeout_seconds': 600})
    ScriptedAPI.commands = [
        ('bash', {'command': "printf 'print(123)' > /tmp/check.py"}),
        ('bash', {'command': 'python3 /tmp/check.py'}),
        ('bash', {'command': "printf 'def invalid(:' > miniapp.py"}),
        ('bash', {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}),
        ('bash', {'command': 'git checkout -- miniapp.py'}),
        ('bash', {'command': "python3 -c \"from pathlib import Path; p=Path('miniapp.py'); p.write_text(p.read_text().replace('a-b','a+b'))\" && cp miniapp.py miniapp.py.backup"}),
        ('bash', {'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}),
    ]
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade',
        lambda metadata,log: {'resolved': 'hidden-ok' in log.read_text(), 'parsed': True})
    directory = tmp_path / 'mini-run'
    def grade_after_submit(*args, **kwargs):
        messages = json.loads((directory/'conversation.json').read_text())
        assert messages[-1]['extra']['exit_status'] == 'Submitted'
        assert any('Submission was not accepted' in str(m.get('content')) for m in messages)
        assert any('123' in str(m.get('content')) for m in messages if m['role'] == 'tool')
        assert not (directory/'workspace/hidden_spec.py').exists()
        return grade_hidden(*args, **kwargs)
    result = run_harness_task(task, cfg, directory, counter=lambda m,t:100,
                             api_factory=ScriptedAPI, grader=grade_after_submit)
    assert result['resolved'], result
    assert result['usage']['code_model_calls'] == 7
    assert result['changed_files'] == ['miniapp.py']
    assert result['validation']['delivery']['passed']
    assert result['excluded_new_backup_files'] == ['miniapp.py.backup']
    assert (directory/'verification/hidden_spec.py').exists()


@pytest.mark.sandbox
@pytest.mark.parametrize('row', SELECTED, ids=[row['task_id'] for row in SELECTED])
def test_selected_actual_repository_import_and_render(row, tmp_path):
    task = Task.from_file(Path(row['task_file']))
    tree = {'max_files': task.runtime.max_repo_files, 'max_bytes': task.runtime.max_repo_bytes,
            'allowed_symlinks': task.runtime.allowed_symlinks}
    baseline, workspace = prepare_repository(Path(task.repo), tmp_path/'actual', **tree)
    sandbox = create_sandbox(workspace, Events(tmp_path/'events.jsonl'), task.protected_paths, 45, baseline, task.runtime)
    sandbox.enable_agent_session()
    package = next(name for name in ('marshmallow','astroid','pyvista','pvlib','pydicom')
                   if (workspace/name/'__init__.py').exists() or (workspace/'src'/name/'__init__.py').exists())
    display = None
    try:
        command = ['python3', '-c', 'import ' + package + '; print("actual-import-ok")']
        if package == 'pyvista':
            cfg = json.loads((ROOT/'configs/mini-session-dev5.json').read_text())
            display = PublicDisplay(cfg['xvfb_executable'], tmp_path)
            sandbox.public_display = display
            command = ['python3','-c','import pyvista as pv; p=pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); p.show(auto_close=True); print("actual-import-ok")']
        result = sandbox.run(command)
        assert result['returncode'] == 0 and 'actual-import-ok' in result['output'], result
    finally:
        sandbox.cancel()
        if display:
            display.close()


def test_five_case_selection_preserves_declared_order_and_rejects_duplicates(tmp_path):
    sys.path.insert(0, str(ROOT/'scripts'))
    from types import SimpleNamespace

    from run_harness_compare import prepare
    prior, output = tmp_path/'prior', tmp_path/'new'
    prior.mkdir()
    tasks = [{'task_id': str(i)} for i in range(10)]
    (prior/'prepared.json').write_text(json.dumps({'protocol':'hidden_reference_tests_harness_v1','tasks':tasks}))
    args = SimpleNamespace(output=output, reuse_prepared=prior, limit=5, arms=['mini'], task_ids='0,3,4,7,8')
    prepare(SimpleNamespace(update=lambda **kw:None),args)
    assert [r['task_id'] for r in json.loads((output/'prepared.json').read_text())['tasks']] == ['0','3','4','7','8']
    args.task_ids = '0,0,4,7,8'
    with pytest.raises(ValueError):
        prepare(SimpleNamespace(update=lambda **kw:None),args)
