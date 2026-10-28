import fcntl
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_harness_compare import old_input as inherited_old_input

from screenshot_agent.config import RuntimeConfig, Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.harness import PROTOCOL
from screenshot_agent.hidden_eval import GRADER_STARTED, grade_hidden, prepare_hidden_task
from screenshot_agent.repository import export_patch, prepare_repository
from screenshot_agent.runtime import OCISandbox
from screenshot_agent.scale_data import sha256, tree_hash
from screenshot_agent.workflow import RepairWorkflow

old_input = inherited_old_input
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('regrade_workflow', ROOT / 'scripts/regrade_workflow.py')
regrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regrade)


def test_partial_cohort_excludes_active_and_keeps_frozen_membership(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    tasks = [{'task_id': name} for name in ('finished', 'active', 'later')]
    for row in tasks:
        folder = source / 'runs' / row['task_id']
        folder.mkdir(parents=True)
        (folder / 'workflow.lock').touch()
        atomic_json(folder / 'result.json', {'status': 'unresolved'})
        atomic_json(folder / 'workflow.json', {'status': 'unresolved' if row['task_id'] != 'later' else 'running'})
    with (source / 'runs/active/workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = regrade.completed_selection(source, output, tasks)
    assert selected == [tasks[0]]
    atomic_json(output / 'plan.json', {'selection_mode': 'completed_only', 'task_ids': ['finished'], 'source_total': 3})
    atomic_json(source / 'runs/later/workflow.json', {'status': 'unresolved'})
    assert regrade.completed_selection(source, output, tasks) == selected
    report = regrade.summarize(output, selected)
    assert report['total'] == 1 and report['source_total'] == 3 and report['selection_mode'] == 'completed_only'


@pytest.mark.sandbox
def test_readonly_oci_grader_mount_executes_without_modifying_rootfs(tmp_path):
    root = tmp_path / 'rootfs'
    for name in ('usr/bin', 'bin', 'lib', 'lib64', 'proc', 'dev', 'tmp', 'root', 'testbed', 'workspace', 'grader'):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / 'usr/bin/python3').touch()
    (root / 'sca_worker.py').touch()
    repo = tmp_path / 'repo'
    repo.mkdir()
    script = tmp_path / 'eval.sh'
    script.write_text("printf 'grader-ok\\n'; if echo changed >> /grader/eval.sh; then exit 9; fi\n")
    original_script = script.read_bytes()
    sandbox = OCISandbox(repo, Events(tmp_path / 'events'), timeout=10,
                        runtime=RuntimeConfig(backend='oci', rootfs=str(root), python='/usr/bin/python3'))

    def with_host_binaries(command):
        # Only supply shared system executables to this minimal synthetic OCI root.
        additions = []
        for name in ('/usr', '/bin', '/lib', '/lib64'):
            if Path(name).exists():
                additions += ['--ro-bind', name, name]
        index = command.index('--chdir')
        return command[:index] + additions + command[index:]

    command = with_host_binaries(sandbox.command(['/bin/bash', '/grader/eval.sh'], {'/grader/eval.sh': script}))
    index = command.index('/grader') - 1
    assert command[index:index + 2] == ['--tmpfs', '/grader']
    old_command = command[:index] + command[index + 2:]
    old = subprocess.run(old_command, capture_output=True, text=True, timeout=15)
    assert old.returncode and "Can't create file at /grader/eval.sh: Read-only file system" in old.stderr
    fixed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert fixed.returncode == 0 and 'grader-ok' in fixed.stdout, fixed.stderr
    assert 'Read-only file system' in fixed.stderr
    assert not list((root / 'grader').iterdir())
    assert script.read_bytes() == original_script
    agent_command = with_host_binaries(sandbox.command(['/bin/bash', '-c', 'test ! -e /grader/eval.sh']))
    assert subprocess.run(agent_command, timeout=15).returncode == 0
    with pytest.raises(ValueError, match='/grader'):
        sandbox.command(['/bin/true'], {'/grader/../testbed/evil': script})


def test_grader_bootstrap_error_is_not_a_wrong_answer(old_input, tmp_path, monkeypatch):
    folder = tmp_path / 'prepared'
    prepare_hidden_task(old_input / 'task.json', folder)
    task = Task.from_file(folder / 'task.json')
    monkeypatch.setattr('screenshot_agent.hidden_eval.create_sandbox', lambda *a, **k: SimpleNamespace(
        run=lambda *a, **k: {'output': "bwrap: Can't create file at /grader/eval.sh: Read-only file system\n",
                            'returncode': 1, 'timed_out': False}, cancel=lambda: None))
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade',
                        lambda *a: pytest.fail('Never score a test suite that could not start'))
    value = grade_hidden(task, Path(task.repo), Events(tmp_path / 'events'), tmp_path / 'log')
    assert value['runtime_failure'] and not value['resolved'] and not value['grader_started']
    assert value['error'] == 'hidden_grader_start_failed'


def test_workflow_marks_grading_environment_failure_as_error(tmp_path, monkeypatch):
    monkeypatch.setattr('screenshot_agent.workflow.export_patch', lambda *a, **k: {'patch_bytes': 1})
    monkeypatch.setattr('screenshot_agent.workflow.replay_patch', lambda *a, **k: {'applied': True})
    flow = object.__new__(RepairWorkflow)
    flow.directory = flow.baseline = flow.workspace = tmp_path
    flow.tree = {}
    flow.save = flow.phase = lambda *a, **k: None
    flow.events = Events(tmp_path / 'events')
    flow.task = SimpleNamespace(benchmark='private', protected_paths=[])
    flow.config = SimpleNamespace(limits=SimpleNamespace(command_timeout_seconds=5))
    flow.grader = lambda *a, **k: {'resolved': False, 'runtime_failure': True}
    result = flow.finalize()
    assert result['status'] == 'error' and result['runtime_failure']


@pytest.mark.sandbox
def test_saved_patch_regrading_preserves_inputs_and_resumes(old_input, tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared'
    row = prepare_hidden_task(old_input / 'task.json', prepared)
    task = Task.from_file(prepared / 'task.json')
    source = tmp_path / 'source'
    original = source / 'runs' / task.task_id
    original.mkdir(parents=True)
    baseline, workspace = prepare_repository(Path(task.repo), original)
    (workspace / 'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    export_patch(baseline, workspace, original / 'patch.diff', [])
    (original / 'workflow.lock').touch()
    atomic_json(original / 'workflow.json', {'status': 'unresolved', 'baseline_hash': tree_hash(baseline)})
    atomic_json(original / 'config.json', {'limits': {'command_timeout_seconds': 10}})
    atomic_json(original / 'result.json', {'task_id': task.task_id, 'protocol': PROTOCOL,
                                         'status': 'unresolved', 'resolved': False})
    before = {str(path): sha256(path) for path in original.glob('*.json')}
    before['patch'] = sha256(original / 'patch.diff')
    monkeypatch.setattr('screenshot_agent.hidden_eval.official_grade', lambda meta, log: {
        'resolved': 'hidden-ok' in log.read_text(), 'parsed': True})
    output = tmp_path / 'regrade'
    result = regrade.regrade_one(source, output, row)
    assert result['resolved'] and result['status'] == 'verified' and result['model_requests'] == 0
    assert GRADER_STARTED in (Path(result['attempt']) / 'test.log').read_text()
    assert tree_hash(baseline) == json.loads((original / 'workflow.json').read_text())['baseline_hash']
    assert sha256(original / 'patch.diff') == before.pop('patch')
    assert {str(path): sha256(path) for path in original.glob('*.json')} == before
    assert not (workspace / 'hidden_spec.py').exists()
    assert regrade.regrade_one(source, output, row) == result
    summary = regrade.summarize(output, [row])
    assert summary['valid_report'] and summary['resolved'] == 1
    (original / 'patch.diff').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        regrade.regrade_one(source, output, row)
