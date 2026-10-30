import json
import sys
from pathlib import Path

import pytest
from test_repair_workflow import basic_task, config
from test_workflow_feedback_v2 import flow as inherited_flow
from test_workflow_feedback_v2 import register, result

from screenshot_agent.repository import export_patch
from screenshot_agent.workflow import run_workflow_task
from screenshot_agent.workflow_demo import DemoAPI, patch
from screenshot_agent.workflow_feedback import compare_public
from screenshot_agent.workflow_support import patch_excerpt, public_test_files, select_public_tests

flow = inherited_flow


def linear_config():
    cfg = config()
    return cfg.model_copy(update={'workflow': cfg.workflow.model_copy(update={'execution_mode': 'linear'})})


def test_package_tests_and_unittest_names_are_discovered(tmp_path):
    for name in ['pvlib/tests/test_iam.py', 'pydicom/tests/test_charset.py', 'astroid/tests/unittest_inference.py']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('def test_fresnel_iam():\n assert True\n')
    (tmp_path / '.sca_scratch').mkdir()
    (tmp_path / '.sca_scratch/test_fake.py').write_text('assert False')
    assert len(public_test_files(tmp_path)) == 3
    selected = select_public_tests(tmp_path, 'Fix fresnel iam in pvlib', 2)
    assert len(selected) == 2 and selected[0] == 'pvlib/tests/test_iam.py::test_fresnel_iam'


def test_param_id_drift_is_uncertain_not_a_regression_or_pass():
    old = result({'test_time[07:11:55 2026-10-28]': 'passed', 'stable': 'passed'}, 0)
    new = result({'test_time[07:32:29 2026-10-28]': 'passed', 'stable': 'passed'}, 0)
    assert compare_public(old, new)['reason'] == 'unstable_parameter_ids'
    new['cases']['stable'] = 'failed'
    assert compare_public(old, new)['status'] == 'failed'
    assert compare_public(result({'test_size[1]': 'passed'}, 0),
                          result({'test_size[2]': 'passed'}, 0))['status'] == 'failed'


def test_source_diff_cannot_be_displaced_by_debug_files(tmp_path):
    (tmp_path / 'package').mkdir()
    (tmp_path / 'package/api.py').write_text('old')
    text = ('diff --git a/debug_aaa.py b/debug_aaa.py\n+' + 'noise' * 8000 + '\n'
            'diff --git a/package/api.py b/package/api.py\n-old\n+REAL_FIX\n')
    excerpt = patch_excerpt(text, tmp_path, 2000)
    assert 'REAL_FIX' in excerpt and len(excerpt) <= 2000
    assert 'debug_aaa.py' in excerpt and 'noise' not in excerpt


@pytest.mark.sandbox
def test_failed_registration_retains_successful_assertion(flow):
    assert register(flow)['baseline_reproduced']
    original = flow.state['reproduction']['sha256']
    (flow.workspace / 'broken.py').write_text('print("Test FAILED")\n')
    answer = flow.feedback.register_reproduction('broken.py', 'wrong')
    assert not answer['baseline_reproduced'] and answer['previous_verified_reproduction_retained']
    assert flow.state['reproduction']['sha256'] == original
    assert flow.feedback.check('still broken')['reproduction']['status'] == 'failed'


@pytest.mark.sandbox
def test_new_invalid_registration_cannot_erase_a_candidate_failure(flow):
    assert register(flow)['baseline_reproduced']
    (flow.workspace / 'broken.py').write_text('print("ok")\n')
    flow.feedback.register_reproduction('broken.py', 'wrong')
    checked = flow.feedback.check('not fixed')
    assert checked['reproduction']['status'] == 'failed' and not checked['passed']


@pytest.mark.sandbox
def test_linear_loop_has_no_planner_or_reproducer_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(DemoAPI, 'script', [
        ('read_file', {'file_path': 'maths.py'}),
        ('apply_patch', {'file_path': 'maths.py', 'patch': patch('return a-b', 'return a+b')}),
        'Implemented and checked addition.'])
    folder = tmp_path / 'run'
    outcome = run_workflow_task(basic_task(tmp_path), linear_config(), folder, reader_name='ocr',
                                api_factory=DemoAPI, counter=lambda m, t: 100, model_inference=False)
    assert outcome['resolved'], outcome
    assert outcome['usage']['code_model_calls'] == 3 and outcome['retry_count'] == 0
    state = json.loads((folder / 'workflow.json').read_text())
    assert [p['phase'] for p in state['plan']] == ['repair']


@pytest.mark.sandbox
def test_source_patch_omits_scratch_and_hypothesis_artifacts(flow):
    for directory in ['.sca_scratch', '.hypothesis']:
        (flow.workspace / directory).mkdir()
        (flow.workspace / directory / 'test_debug.py').write_text('print("debug")')
    (flow.workspace / 'maths.py').write_text('def add(a,b): return a+b\n')
    exported = export_patch(flow.baseline, flow.workspace, flow.directory / 'patch.diff', [])
    assert exported['changed_files'] == ['maths.py']


@pytest.mark.sandbox
def test_quick_check_detects_broken_import_without_required_reproduction(flow):
    # Existing package is snapshotted into both baseline and workspace for this check.
    for root in [flow.baseline, flow.workspace]:
        (root / 'pkg').mkdir()
        (root / 'pkg/__init__.py').write_text('from .api import value\n')
        (root / 'pkg/api.py').write_text('value = 1\n')
    from screenshot_agent.repository import git
    git(flow.baseline, 'add', '-A')
    git(flow.baseline, 'commit', '-qm', 'fixture package')
    git(flow.workspace, 'add', '-A')
    git(flow.workspace, 'commit', '-qm', 'fixture package')
    (flow.workspace / 'pkg/api.py').write_text('renamed = 1\n')
    flow.config = linear_config()
    checked = flow.feedback.check('missing export')
    assert checked['quick_check']['status'] == 'failed'
    assert 'ImportError' in checked['quick_check']['candidate_output']
    assert checked['retry_needed'] and not checked['passed']


def test_smoke_default_and_old_report_compatibility(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    from run_repair_workflow import parser, prepared_tasks, report_method
    assert parser().parse_args(['all']).limit == 10
    (tmp_path / 'prepared.json').write_text(json.dumps({
        'protocol': 'hidden_reference_tests_harness_v1', 'task_count': 10,
        'tasks': [{'task_id': str(i)} for i in range(10)]}))
    assert len(prepared_tasks(tmp_path)) == 10
    (tmp_path / 'workflow_plan.json').write_text(json.dumps({'method': 'corecoder_retrieval_workflow_v2'}))
    assert report_method(tmp_path) == 'corecoder_retrieval_workflow_v2'
    data = json.loads((tmp_path / 'prepared.json').read_text())
    data['tasks'].append({'task_id': 'unexpected'})
    (tmp_path / 'prepared.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepared_tasks(tmp_path)


@pytest.mark.sandbox
def test_owned_public_display_releases_process(tmp_path):
    from screenshot_agent.public_display import PublicDisplay
    executable = Path('/data/zjr/screenshot_code_agent/workflow_v3_runtime/xvfb_extract/usr/bin/Xvfb')
    if not executable.exists():
        pytest.skip('Optional local headless dependency is not installed')
    display = PublicDisplay(executable, tmp_path)
    assert display.proc.poll() is None and display.socket.is_socket()
    display.close()
    assert display.proc.poll() is not None
    display.close()


def test_public_sampling_uses_collected_ids_and_bounds_parameterized_cases():
    from types import SimpleNamespace

    from screenshot_agent.workflow_support import PUBLIC_PLUGIN
    scope = {}
    exec(PUBLIC_PLUGIN.replace('SELECTED_NODEIDS = None', "SELECTED_NODEIDS = ['a.py::test_ok']"), scope)
    items = [SimpleNamespace(nodeid=f'a.py::test_ok[{i}]') for i in range(100)]
    items += [SimpleNamespace(nodeid='a.py::unrelated')]
    skipped = []
    cfg = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: skipped.extend(items)))
    scope['pytest_collection_modifyitems'](cfg, items)
    assert len(items) == 3 and len(skipped) == 98


def test_smoke_preflight_refuses_wrong_code_inputs_or_missing_receipt(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_repair_workflow as runner

    cfg = linear_config()
    task_file = tmp_path / 'input.json'
    task_file.write_text('{}')
    tasks = [{'task_id': 'one', 'task_file': str(task_file)}]
    record = {'ready': True, 'implementation': runner.implementation(), 'config': cfg.model_dump(),
              'tasks': [{'task_id': 'one', 'task_sha256': runner.sha256(task_file)}]}
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(record))
    runner.validate_smoke_preflight(tasks, cfg, receipt)
    with pytest.raises(ValueError, match='requires'):
        runner.validate_smoke_preflight(tasks, cfg, None)
    task_file.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='same fixed'):
        runner.validate_smoke_preflight(tasks, cfg, receipt)
    monkeypatch.setattr(runner, 'implementation', lambda: {'new': 'code'})
    with pytest.raises(ValueError, match='code/config'):
        runner.validate_smoke_preflight(tasks, cfg, receipt)


@pytest.mark.sandbox
def test_actual_oci_software_render_uses_owned_display_and_system_cpp(tmp_path):
    from screenshot_agent.config import Task
    from screenshot_agent.control import Events
    from screenshot_agent.public_display import PublicDisplay
    from screenshot_agent.runtime import create_sandbox

    task_path = Path('/data/zjr/screenshot_code_agent/harness_compare_20261027/data/tasks/'
                     'pyvista__pyvista-4329/task.json')
    executable = Path('/data/zjr/screenshot_code_agent/workflow_v3_runtime/xvfb_extract/usr/bin/Xvfb')
    if not task_path.exists() or not executable.exists():
        pytest.skip('Prepared local OCI rendering dependencies are not available')
    repo = tmp_path / 'repo'
    repo.mkdir()
    runtime = Task.from_file(task_path).runtime
    display = PublicDisplay(executable, tmp_path)
    sandbox = create_sandbox(repo, Events(tmp_path/'events.jsonl'), [], 30, repo, runtime)
    sandbox.public_display = display
    try:
        rendered = sandbox.run("python -u -c 'import vtk; w=vtk.vtkXOpenGLRenderWindow(); "
                               "w.SetOffScreenRendering(1); w.Render(); print(\"render-ok\")'")
        assert rendered['returncode'] == 0 and 'render-ok' in rendered['output'], rendered
    finally:
        sandbox.cancel()
        display.close()
