import ast
import json
import sys
import time
from pathlib import Path

import pytest
from test_repair_workflow import config
from test_workflow_feedback_v2 import flow as inherited_flow

from screenshot_agent.control import Events
from screenshot_agent.workflow_cases import CASES, development_case
from screenshot_agent.workflow_config import PhaseEnded, WorkflowBudget
from screenshot_agent.workflow_demo import DemoAPI, patch
from screenshot_agent.workflow_support import classify_changes, select_public_tests

flow = inherited_flow


def development_config():
    cfg = config()
    return cfg.model_copy(update={'workflow': cfg.workflow.model_copy(update={
        'execution_mode': 'linear', 'development_casebook': True, 'require_issue_verification': True})})


def fake_contract(monkeypatch, task_id):
    monkeypatch.setitem(CASES, task_id, ('Addition must produce the correct sum.',
                                      'from maths import add\nassert add(2, 3) == 5, "wrong sum"\n'))


def test_changed_files_keep_real_new_modules_and_exclude_diagnostics(tmp_path):
    (tmp_path / 'debugger.py').write_text('existing module')
    changes = classify_changes(tmp_path, ['debugger.py', 'debug_run.py', 'test_fix.py',
                                           'package/new_feature.py', '.scratch.py', 'repro.py'])
    assert changes['implementation'] == ['debugger.py', 'package/new_feature.py']
    assert changes['diagnostics'] == ['debug_run.py', 'test_fix.py', '.scratch.py', 'repro.py']


def test_contracts_are_explicit_opt_in_and_compatible_with_old_python():
    assert not config().workflow.development_casebook
    assert len(CASES) == 9
    for task_id in CASES:
        case = development_case(task_id)
        ast.parse(case['script'], feature_version=(3, 6))
        assert case['scope'].startswith('known development regression')
    assert development_case('pylint-dev__astroid-984') is None


def test_known_public_module_selection_excludes_unrelated_slow_modules(tmp_path):
    for name in ('test_iam.py', 'test_solarposition.py'):
        (tmp_path / name).write_text('def test_physical():\n    assert True\n')
    chosen = select_public_tests(tmp_path, 'physical IAM solarposition', 12, allowed_files=['test_iam.py'])
    assert chosen == ['test_iam.py::test_physical']


@pytest.mark.sandbox
def test_debug_only_patch_cannot_pass_and_cannot_be_delivered(flow):
    flow.config = development_config()
    (flow.workspace / 'debug_run.py').write_text('print("passed")\n')
    result = flow.feedback.check('debug only')
    assert result['regression']['status'] == 'passed'
    assert not result['non_reproduction_changed_files']
    assert result['retry_needed'] and not result['passed']
    flow.finalize()
    assert not (flow.directory / 'patch.diff').read_text()
    assert (flow.workspace / '.sca_scratch/delivery_cleanup/debug_run.py').exists()


@pytest.mark.sandbox
def test_known_issue_retries_even_when_public_regressions_pass(flow, monkeypatch):
    flow.config = development_config()
    fake_contract(monkeypatch, flow.task.task_id)
    flow.api_factory = DemoAPI
    monkeypatch.setattr(DemoAPI, 'script', [
        ('apply_patch', {'file_path': 'maths.py', 'patch': patch('return a-b', 'return a-b+0')}),
        'Finished; public tests pass.',
        ('apply_patch', {'file_path': 'maths.py', 'patch': patch('return a-b+0', 'return a+b')}),
        'Fixed the original issue.'])
    flow.repair()
    assert flow.state['retry_count'] == 1
    first, last = flow.state['checks'][0], flow.state['checks'][-1]
    assert first['regression']['status'] == 'passed'
    assert first['known_issue']['status'] == 'failed' and first['retry_needed']
    assert last['known_issue']['status'] == 'passed' and last['passed']
    assert flow.budget.code_calls == 4
    assert 'wrong sum' in flow.state['reviews'][0]['observed']['known_issue']['output']


def test_phase_time_reserve_is_available_to_final_checks(tmp_path, monkeypatch):
    cfg = config()
    budget = WorkflowBudget(cfg, Events(tmp_path / 'events'), lambda m, t: 100, tmp_path / 'budget.json')
    budget.phase_wall_reserve = 150
    now = time.monotonic()
    monkeypatch.setattr('screenshot_agent.workflow_config.time.monotonic', lambda: now)
    budget.started = now - cfg.limits.max_wall_seconds + 160
    assert budget.remaining_seconds() == 10
    budget.started -= 20
    with pytest.raises(PhaseEnded, match='final checks'):
        budget.remaining_seconds()
    with pytest.raises(PhaseEnded, match='final checks'):
        budget.reserve('code', cfg.code, [])
    budget.phase_wall_reserve = 0
    assert budget.remaining_seconds() == 140


def test_v4_default_does_not_break_reporting_old_v3_results(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_repair_workflow import parser, report_method
    assert parser().parse_args(['all']).config.name == 'workflow-v4.json'
    (tmp_path / 'workflow_plan.json').write_text(json.dumps({'method': 'corecoder_repair_workflow_v3'}))
    assert report_method(tmp_path) == 'corecoder_repair_workflow_v3'
