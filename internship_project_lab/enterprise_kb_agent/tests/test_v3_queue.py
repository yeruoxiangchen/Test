import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest


def queue():
    path = Path(__file__).resolve().parents[1] / 'scripts/eval_v3.py'
    spec = importlib.util.spec_from_file_location('suite_v3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v3_queue_records_failed_stage_and_can_show_status(tmp_path, monkeypatch, capsys):
    q = queue()
    calls = []
    def fail(root, stage, args):
        calls.append(args)
        raise RuntimeError('service down')
    monkeypatch.setattr(q, 'run_command', fail)
    monkeypatch.setattr(sys, 'argv', ['eval_v3.py', 'work', '--output', str(tmp_path), '--arms', 'fixed-bm25'])
    with pytest.raises(RuntimeError, match='service down'):
        q.main()
    state = json.loads((tmp_path / 'queue_state.json').read_text())
    assert state['status'] == 'failed' and state['stage'] == 'fixed-bm25'
    assert '--split' in calls[0] and calls[0][calls[0].index('--split')+1] == 'dev'
    monkeypatch.setattr(sys, 'argv', ['eval_v3.py', 'status', '--output', str(tmp_path)])
    q.main()
    assert 'service down' in capsys.readouterr().out


def test_v3_protocol_change_fails_before_any_inference(tmp_path, monkeypatch):
    q = queue()
    (tmp_path / 'protocol.json').write_text('{"source_fingerprint":"different"}')
    def forbidden(*args):
        raise AssertionError('must not run')
    monkeypatch.setattr(q, 'run_command', forbidden)
    monkeypatch.setattr(sys, 'argv', ['eval_v3.py', 'work', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='changed'):
        q.main()


def report_fixture(tmp_path):
    protocol = {'arms': ['fixed-bm25'], 'limit': 1, 'split': 'dev',
                'source_fingerprint': 'original-inference-code', 'queue_sha256': 'original-queue-code'}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    arm = tmp_path/'fixed-bm25'
    arm.mkdir()
    (arm/'config.json').write_text(json.dumps({'arm':'fixed-bm25', 'split':'dev', 'limit':1,
        'case_ids':['a'], 'dataset_receipt':{}, 'source_fingerprint':protocol['source_fingerprint']}))
    (arm/'predictions.jsonl').write_text('{"id":"a"}\n')
    (arm/'grading.jsonl').write_text('{"id":"a"}\n')
    (arm/'report.json').write_text(json.dumps({'metrics':{}, 'records':1, 'expected':1,
        'model_calls':1, 'prompt_tokens':10, 'latency_seconds':{'p50':.1}}))
    return protocol


def test_v3_report_only_does_not_run_inference(tmp_path, monkeypatch):
    q = queue()
    original = report_fixture(tmp_path)
    def forbidden(*args):
        raise AssertionError('report-only must not run inference')
    monkeypatch.setattr(q, 'run_command', forbidden)
    monkeypatch.setattr(q, 'export_review', lambda *a: None)
    monkeypatch.delenv('KB_JUDGE_BASE_URL', raising=False)
    monkeypatch.setattr(sys, 'argv', ['eval_v3.py', 'report', '--output', str(tmp_path), '--arms', 'fixed-bm25', '--limit', '1'])
    q.main()
    state = json.loads((tmp_path / 'queue_state.json').read_text())
    assert state['status'] == 'complete' and state['semantic'].startswith('pending_independent_judge')
    assert (tmp_path / 'report.md').exists()
    assert json.loads((tmp_path/'protocol.json').read_text()) == original
    receipt = json.loads((tmp_path/'reporting_manifest.json').read_text())
    assert receipt['original_inputs_unchanged']
    assert receipt['reporting_protocol']['source_fingerprint'] != original['source_fingerprint']


@pytest.mark.parametrize('corruption', ['identity', 'duplicates', 'cases', 'dataset', 'limit'])
def test_report_recovery_rejects_mixed_or_incomplete_originals(tmp_path, corruption):
    q = queue()
    protocol = report_fixture(tmp_path)
    cfg_path = tmp_path/'fixed-bm25/config.json'
    cfg = json.loads(cfg_path.read_text())
    if corruption == 'identity':
        cfg['source_fingerprint'] = 'new-inference-code'
    elif corruption == 'duplicates':
        (tmp_path/'fixed-bm25/predictions.jsonl').write_text('{"id":"a"}\n{"id":"a"}\n')
    elif corruption == 'cases':
        cfg['case_ids'] = ['a','b']
    elif corruption == 'limit':
        cfg['limit'] = 2
    else:
        second = tmp_path/'agent-bm25'
        second.mkdir()
        other = {**cfg,'arm':'agent-bm25','dataset_receipt':{'different':'dataset'}}
        (second/'config.json').write_text(json.dumps(other))
        protocol['arms'].append('agent-bm25')
        (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        q.report_inputs(tmp_path, protocol)
