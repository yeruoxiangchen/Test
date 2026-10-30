import json
from types import SimpleNamespace

import pytest

from screenshot_agent.control import NoProgress
from screenshot_agent.mini_actions import execute_action
from screenshot_agent.mini_repair_control import RepairControl


def setup_control():
    delivery = SimpleNamespace(last_response={'passed': False, 'output': 'missing required parameter'},
                               sandbox=SimpleNamespace(events=SimpleNamespace(emit=lambda *a, **k: None)))
    return RepairControl(delivery)


def result(text='', code=0, timed=False):
    return {'returncode': code, 'output': text, 'timed_out': timed}


def test_unchanged_failed_checks_stop_but_real_edit_resets():
    c = setup_control()
    for _ in range(6):
        c.before({'tool': 'check_patch'})
        c.after({'tool': 'check_patch'}, result('failed', 1))
    assert 'missing required parameter' in c.message()['content']
    with pytest.raises(NoProgress):
        c.before({'tool': 'check_patch'})
    c.after({'tool': 'edit_file'}, result('diff --git changed'))
    assert c.before({'tool': 'check_patch'}) is None
    c.delivery.last_response = {'passed': True, 'output': 'passed'}
    c.after({'tool': 'check_patch'}, result())
    assert c.message() is None and c.stalls == 0


def test_timeout_repeat_blocked_until_actual_edit_and_reads_allowed():
    c = setup_control()
    action = {'tool': 'bash', 'command': 'python3 loop.py'}
    c.after(action, result('', -9, True))
    assert c.before(action)['retry_blocked']
    assert c.before({'command': 'cat source.py'}) is None
    c.after({'tool': 'edit_file'}, result('', 2))
    assert c.before(action)['retry_blocked']
    c.after({'tool': 'edit_file'}, result('diff changed'))
    assert c.before(action) is None


def test_echo_loop_cannot_claim_success():
    c = setup_control()
    c.after({'tool': 'check_patch'}, result('failed', 1))
    for _ in range(5):
        c.after({'command': 'cd /testbed && echo ALL DONE'}, result('ALL DONE'))
    with pytest.raises(NoProgress):
        c.before({'command': 'echo ALL DONE'})


def test_shell_timeout_cap_and_submit_protocol():
    calls = []
    sb = SimpleNamespace(timeout=120, run=lambda cmd, **kw: calls.append(kw) or result())
    budget = SimpleNamespace(config=SimpleNamespace(mini_repair_control=True), remaining_seconds=lambda: 500)
    execute_action({'command': 'python3 reproduce.py'}, sb, budget, None)
    execute_action({'command': 'python3 -m pytest tests'}, sb, budget, None)
    assert [c['timeout'] for c in calls] == [30, 120]
    assert execute_action({'tool': 'submit'}, sb, budget, None)['output'].startswith('COMPLETE_TASK_AND_SUBMIT')


def test_state_is_in_actual_next_model_request(tmp_path):
    from test_harness_compare import ScriptedAPI, configuration

    from screenshot_agent.control import Events
    from screenshot_agent.harness import HarnessBudget, load_mini, make_mini

    cfg = configuration('mini').model_copy(update={'mini_session': True, 'mini_verified': True,
                                                  'mini_repair_control': True})
    load_mini(tmp_path)
    events = Events(tmp_path/'events.jsonl')
    budget = HarnessBudget(cfg, events, lambda m,t:100)
    ScriptedAPI.commands = [('bash', {'command': 'cat source.py'})]
    api = ScriptedAPI(cfg.code, budget, events, policy=cfg.agent)
    sb = SimpleNamespace(description=lambda: {'repository_mount': '/testbed'})
    delivery = SimpleNamespace(last_response=None)
    agent = make_mini(cfg, budget, events, sb, tmp_path, api=api, submission_check=delivery)
    agent.env.control.failed = 'SetCells overload is invalid'
    agent.model.query([{'role': 'user', 'content': 'fix'}])
    assert 'SetCells overload is invalid' in json.dumps(api.seen[-1])
