"""Exercise failed dev5 delivery and rollback on real frozen task runtimes."""
import json
from pathlib import Path

import pytest
from test_harness_compare import configuration
from test_mini_session import ROOT, SELECTED
from test_mini_verified import checked_session as inherited_session
from test_mini_verified import runtime as inherited_runtime
from test_mini_verified import synthetic_repo_does_not_load_unrelated_plotting_plugin as inherited_plugins

from screenshot_agent.config import Task
from screenshot_agent.control import Events, NoProgress
from screenshot_agent.harness import HarnessBudget, MiniEnvironment, MiniModel
from screenshot_agent.harness_runner import install_development_case
from screenshot_agent.mini_recovery import RecoveryDelivery, behavior_cases
from screenshot_agent.mini_semantic_control import SemanticControl
from screenshot_agent.public_display import PublicDisplay
from screenshot_agent.repository import git, prepare_repository
from screenshot_agent.runtime import create_sandbox

checked_session = inherited_session
runtime = inherited_runtime
synthetic_repo_does_not_load_unrelated_plotting_plugin = inherited_plugins

SCRIPT = '''
import json
from miniapp import add
cases = {'add': {'passed': add(2,3)==5}, 'other': {'passed': add(3,3)==6}}
print('__SCA_BEHAVIOR__' + json.dumps(cases))
raise SystemExit(0 if all(r['passed'] for r in cases.values()) else 1)
'''


def recovery(checked_session):
    sb, budget, old = checked_session
    budget.config.mini_repair_control = budget.config.mini_recovery = True
    for root in (old.baseline, old.workspace):
        (root/old.case['path']).write_text(SCRIPT)
    return sb, budget, RecoveryDelivery(old.baseline, old.workspace, old.directory, old.protected,
                                       old.tree, sb, budget, old.case, runtime=old.runtime)


@pytest.mark.sandbox
def test_debug_artifact_does_not_hide_behavior_and_partial_survives_checkout(checked_session):
    sb, _, d = recovery(checked_session)
    d.initialize()
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return a+b if a==2 else a-b\n')
    (sb.repo/'reproduce_issue.py').write_text('')
    checked = d()
    assert not checked['passed'] and 'other' in checked['output']
    assert 'Move reproduction' not in checked['output']
    assert d.partial and not d.best
    report = json.loads((d.directory/'delivery_check.json').read_text())
    assert report['visible_development_case'] is not None
    assert report['comparison']['status'] == 'passed'
    assert report['export']['excluded_new_debug_files'] == ['reproduce_issue.py']
    assert sb.run('git checkout HEAD -- miniapp.py')['returncode'] == 0
    selected = d.select_final(d.directory/'final.diff')['candidate_selection']
    assert selected['partial'] and not selected['verified'] and selected['used_earlier_candidate']
    assert 'return a+b if a==2' in (d.directory/'final.diff').read_text()
    assert (d.directory/'last_workspace_candidate.diff').read_text() == ''
    assert (sb.repo/'reproduce_issue.py').exists()  # Exclusion does not delete user files.


@pytest.mark.sandbox
def test_regression_cannot_be_retained_and_full_candidate_remains_authoritative(checked_session):
    sb, _, d = recovery(checked_session)
    d.initialize()
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return 5\n')
    assert not d()['passed']
    assert not d.partial and not d.best  # One behavior passes but old test regresses.
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return a+b\n')
    (sb.repo/'reproduce_issue.py').write_text('')
    assert d()['passed']
    (sb.repo/'miniapp.py').write_text('def add(a,b):\n    return a-b\n')
    selected = d.select_final(d.directory/'final.diff')['candidate_selection']
    assert selected['verified'] and not selected['partial'] and selected['used_earlier_candidate']


@pytest.mark.sandbox
def test_comment_reads_reworded_tests_and_rollback_do_not_reset_progress(checked_session):
    sb, budget, d = recovery(checked_session)
    d.initialize()
    c = SemanticControl(d)
    c.feedback({'passed': False, 'output': 'physical n_ar missing\n' + 'verbose pytest log\n' * 400})
    assert 'physical n_ar missing' in c.message()['content']
    c.feedback({'passed': False, 'output': 'Fix other contract'})
    env = MiniEnvironment(sb, budget, sb.events, d, c)
    (sb.repo/'miniapp.py').write_text('# cosmetic\ndef add(a,b):\n    return a-b\n')
    c.after({'tool': 'edit_file'}, {'returncode': 0, 'output': 'diff', 'timed_out': False})
    assert c.unchanged == 1
    env.execute({'command': "python3 -c \"from pathlib import Path; Path('miniapp.py').write_text('def add(a,b):\\n    return a+b\\n')\""})
    assert c.unchanged == 0 and c.epoch == 1  # Shell edits detected too.
    env.execute({'command': 'git checkout HEAD -- miniapp.py'})
    assert c.unchanged == 1 and c.epoch == 1
    # Stay below the auto-check cadence here: focus on source progress policy.
    for i in range(11):
        c.before({'command': f'echo alternative reproduction {i}'})
        c.after({'command': f'echo alternative reproduction {i}'},
                {'returncode': 0, 'output': 'self-test passed', 'timed_out': False})
    with pytest.raises(NoProgress):
        c.guard()
    with pytest.raises(NoProgress):
        MiniModel(None, {}, c).query([])  # Stop before paying for the next API call.


@pytest.mark.sandbox
@pytest.mark.parametrize('row', SELECTED, ids=[r['task_id'] for r in SELECTED])
def test_latest_dev5_patches_and_pyvista_rollback_in_real_runtime(row, tmp_path):
    task = Task.from_file(Path(row['task_file']))
    tree = {'max_files': task.runtime.max_repo_files, 'max_bytes': task.runtime.max_repo_bytes,
            'allowed_symlinks': task.runtime.allowed_symlinks}
    baseline, workspace = prepare_repository(Path(task.repo), tmp_path/'actual', **tree)
    case = install_development_case(task, baseline, workspace, recovery=True)
    protected = [*task.protected_paths, case['path']]
    events = Events(tmp_path/'events.jsonl')
    sb = create_sandbox(workspace, events, protected, 90, baseline, task.runtime)
    sb.enable_agent_session()
    cfg = configuration('mini').model_copy(update={'mini_session': True, 'mini_verified': True,
                                                  'mini_repair_control': True, 'mini_recovery': True})
    budget = HarnessBudget(cfg, events, lambda m,t:100)
    display = None
    if task.task_id.startswith('pyvista'):
        conf = json.loads((ROOT/'configs/mini-verified-dev5.json').read_text())
        display = PublicDisplay(conf['xvfb_executable'], tmp_path)
        sb.public_display = display
    d = RecoveryDelivery(baseline, workspace, tmp_path/'actual', protected, tree, sb, budget, case,
                         runtime=task.runtime, problem=task.problem)
    previous = Path('/data/zjr/screenshot_code_agent/mini_repair_dev5_20261030/mini/runs')/task.task_id
    try:
        d.initialize()
        patch = previous/'patch.diff'
        if task.task_id.startswith('pyvista'):
            # Actual intermediate implementation before the observed git checkout.
            patch = previous/'delivery_checks/3/candidate.diff'
        git(workspace, 'apply', '-', stdin=patch.read_bytes())
        checked = d()
        expected = task.task_id.startswith(('pydicom', 'pylint-dev'))
        assert checked['passed'] == expected, checked
        report = json.loads((d.directory/'delivery_check.json').read_text())
        assert report['visible_development_case'] is not None
        assert behavior_cases(report['public_regression']['behavior'])
        if not expected:
            assert d.best is None
            if task.task_id.startswith('pvlib'):
                # Actual candidate regresses the original missing-parameters
                # ValueError to TypeError. Do not weaken the retention gate.
                assert d.partial is None
                assert report['comparison']['status'] == 'failed'
                assert report['comparison']['lost_passing_tests'] == [
                    'pvlib.tests.test_pvsystem::test_PVSystem_get_iam_interp']
            else:
                assert d.partial is not None
            assert 'behavior_failures' in checked['output']
        if task.task_id.startswith('pyvista'):
            assert sb.run('git checkout HEAD -- pyvista/core/pointset.py')['returncode'] == 0
            selection = d.select_final(d.directory/'final.diff')['candidate_selection']
            assert selection['partial'] and selection['used_earlier_candidate']
            assert 'pyvista/core/pointset.py' in (d.directory/'final.diff').read_text()
    finally:
        sb.cancel()
        if display:
            display.close()
