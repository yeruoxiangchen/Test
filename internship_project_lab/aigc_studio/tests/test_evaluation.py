import json
from pathlib import Path

import pytest

from evaluation.report import aggregate, eligible, percentile, render, timings
from evaluation.support import Monitor, execution_times, save


def record(ident='dataset-i01-1', **changes):
    return {'id': ident, 'phase': 'dataset', 'route': 'studio', 'params': {'mode': 'text-to-image',
            'width': 1024, 'height': 1024, 'steps': 25, 'prompt': '<script>not executable</script>'},
            'accepted': 100, 'claimed': 102, 'execution_start': 103, 'execution_end': 113,
            'output_ready': 115, 'client_received': 116, 'attempt_started': 99,
            'finished': 117, 'success': True, 'worker_id': 'eval-0', 'cached_nodes': [],
            'resources': {'clean_device_observed': True, 'observed_process_peak_mib': 10000}, **changes}


def test_real_execution_events_are_not_claim_timestamps():
    event = {'status': {'messages': [['execution_start', {'timestamp': 103000}],
        ['execution_cached', {'nodes': ['model', 'negative']}], ['execution_success', {'timestamp': 113000}]]}}
    assert execution_times(event) == {'execution_start': 103, 'execution_end': 113, 'cached_nodes': ['model', 'negative']}
    assert execution_times({})['execution_start'] is None
    assert timings(record()) == {'queue_s': 2, 'dispatch_s': 1, 'execution_s': 10,
                                'save_poll_s': 2, 'end_to_end_s': 15, 'client_received_s': 17}


@pytest.mark.parametrize('change', [{'success': False}, {'resumed': True},
    {'resources': {'clean_device_observed': False}}, {'cached_nodes': ['sample']}, {'execution_end': None}])
def test_invalid_measurements_cannot_improve_report(change):
    assert not eligible(record(**change))
    report = aggregate([record(**change)])
    assert sum(g['valid_performance_samples'] for g in report['groups'].values()) == 0
    assert sum(g['completed'] for g in report['groups'].values()) == 1


def test_paired_comparison_requires_both_valid_routes():
    a = record('paired-i01-direct', phase='paired', route='direct', client_received=114)
    b = record('paired-i01-studio', phase='paired', route='studio', client_received=116)
    c = record('paired-i02-studio', phase='paired', route='studio')
    result = aggregate([a, b, c])
    assert result['paired_delta_summary_s']['n'] == 1
    assert result['paired_delta_summary_s']['median'] == 2
    assert percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)


def test_zero_observations_and_other_process_invalidate_memory(tmp_path):
    monitor = Monitor(tmp_path, [{'id': 'eval-0', 'gpu': '0', 'pid': 10}])
    assert monitor.summarize(0, 10, 'eval-0')['clean_device_observed'] is False
    monitor.rows = [{'time': 5, 'gpus': {'0': {'used_mib': 1500, 'processes': [
        {'pid': 10, 'memory_mib': 500}, {'pid': 11, 'memory_mib': 900}]}}, 'rss_mib': {'eval-0': 100}}]
    result = monitor.summarize(0, 10, 'eval-0')
    assert result['observed_process_peak_mib'] == 500
    assert result['other_compute_pids'] == [11]
    assert not result['clean_device_observed']


def test_report_preserves_failures_and_does_not_invent_quality(tmp_path):
    save(tmp_path / 'tasks/a.json', record(output=str(tmp_path / 'media/a.png'), checks=['one teapot']))
    save(tmp_path / 'tasks/b.json', record('dataset-i02-1', success=False, error='GPU out of memory'))
    save(tmp_path / 'state.json', {'phase': 'waiting_for_gpu'})
    render(tmp_path)
    content = (tmp_path / 'report.md').read_text()
    assert '| text-to-image | 40 | 2 | 1 |' in content
    assert 'waiting_for_gpu' in content and '没有导入独立评审结果' in content
    assert 'one_gpu：未完成' in content
    page = (tmp_path / 'review.html').read_text()
    assert '<script>not executable</script>' not in page
    assert '&lt;script&gt;' in page and '未评分' in page
    mapping = json.loads((tmp_path / 'review-mapping.json').read_text())
    assert mapping == {'item-001': 'dataset-i01-1'}


def test_interrupted_throughput_does_not_reuse_completed_batch(tmp_path, monkeypatch):
    from evaluation import run
    evaluation = object.__new__(run.Evaluation)
    evaluation.root = tmp_path
    evaluation.workers = [{'url': 'http://unused', 'id': 'eval-0'}]
    evaluation.protocol = {'timeout_seconds': 1, 'image': {}, 'image_cases': [
        {'id': f'i{i}', 'prompt': 'test'} for i in range(8)]}
    evaluation.clean_device = lambda: None
    monkeypatch.setattr(run, 'request', lambda method, base, path, **kwargs:
                        {'prompt_id': 'p'} if path == '/prompt' else {'p': {'status': {'completed': True}}})
    received = []
    def task(ident, params, phase):
        received.append(ident)
        return {'id': ident, 'success': True, 'resources': {'clean_device_observed': True}}
    evaluation.task = task
    evaluation.throughput('one_gpu')
    first = json.loads((tmp_path / 'one_gpu.json').read_text())
    # Simulate an interrupted prior batch (task records exist but no final window).
    (tmp_path / 'one_gpu.json').unlink()
    evaluation.throughput('one_gpu')
    second = json.loads((tmp_path / 'one_gpu.json').read_text())
    assert first['batch_id'] != second['batch_id']
    assert not set(first['task_ids']) & set(second['task_ids'])
    assert len(received) == 16
