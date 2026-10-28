"""Run the real API/queue against an isolated CPU upstream fixture, including process crashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import httpx

SOURCE = Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    api_root = root / 'api'
    api_root.mkdir(exist_ok=True)
    api_port, fake_port = port(), port()
    api = f'http://127.0.0.1:{api_port}'
    fake = f'http://127.0.0.1:{fake_port}'
    (api_root / 'settings.json').write_text(json.dumps({'workers': [{'id': 'fault-worker', 'url': fake, 'models': ['sdxl'], 'timeout_seconds': 90}]}))
    client = httpx.Client(timeout=10, trust_env=False)
    processes = []
    results = []

    def start(module, service_port, name, environment=None):
        with (root / (name + '.log')).open('a') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'uvicorn', module, '--host', '127.0.0.1', '--port', str(service_port)], cwd=SOURCE, env=environment, stdout=log, stderr=log)
        processes.append(process)
        address = f'http://127.0.0.1:{service_port}'
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited; inspect its log')
            try:
                if client.get(address + '/openapi.json').status_code == 200:
                    return process
            except httpx.HTTPError:
                pass
            time.sleep(.1)
        raise TimeoutError(name + ' failed to start')

    def request(method, base, path, **kwargs):
        response = client.request(method, base + path, **kwargs)
        response.raise_for_status()
        return response.json()

    def control(**values):
        return request('POST', fake, '/control', json=values)

    def submit(prompt, key=None):
        return request('POST', api, '/api/jobs', json={'mode': 'text-to-image', 'prompt': prompt, 'seed': 42, 'width': 512, 'height': 512, 'steps': 6}, headers={'Idempotency-Key': key} if key else {})

    def wait(ident, statuses, limit=35):
        until = time.monotonic() + limit
        while time.monotonic() < until:
            job = request('GET', api, '/api/jobs/' + ident)
            if job['status'] in statuses:
                return job
            time.sleep(.2)
        raise AssertionError(f'Task did not reach {statuses}: {job}')

    def case(name, function):
        begin = time.time()
        try:
            detail = function()
            result = {'case': name, 'passed': True, 'evidence': detail}
        except Exception:
            result = {'case': name, 'passed': False, 'error': traceback.format_exc()}
        result['seconds'] = time.time() - begin
        results.append(result)
        (root / 'results.json').write_text(json.dumps({'kind': 'real API with CPU fault fixture', 'results': results}, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result['passed']:
            raise RuntimeError(name + ' failed')

    environment = {**os.environ, 'AIGC_DATA_ROOT': str(api_root)}
    try:
        start('evaluation.fake_comfy:app', fake_port, 'upstream')
        service = start('backend.app:factory', api_port, 'api', environment)

        def invalid():
            before = request('GET', fake, '/metrics')['submitted']
            response = client.post(api + '/api/jobs', json={'mode': 'text-to-image', 'prompt': 'invalid', 'width': 513})
            assert response.status_code == 422
            assert request('GET', fake, '/metrics')['submitted'] == before
            return {'http_status': response.status_code, 'upstream_submissions': 0}
        case('invalid_input_never_dispatched', invalid)

        def idempotency():
            before = request('GET', fake, '/metrics')['submitted']
            with ThreadPoolExecutor(8) as pool:
                jobs = list(pool.map(lambda _: submit('duplicate', 'same-request'), range(8)))
            assert len({j['id'] for j in jobs}) == 1
            job = wait(jobs[0]['id'], {'succeeded', 'failed'})
            assert job['status'] == 'succeeded'
            assert request('GET', fake, '/metrics')['submitted'] == before + 1
            assert client.get(api + '/api/assets/' + job['outputs'][0]).status_code == 200
            return {'job_id': job['id'], 'concurrent_requests': 8, 'actual_submissions': 1}
        case('concurrent_idempotent_submission', idempotency)

        def cancellation():
            control(delay=12)
            first = submit('cancel-running')
            running = wait(first['id'], {'running'})
            queued = submit('cancel-queued')
            result = request('POST', api, f"/api/jobs/{queued['id']}/cancel")
            assert result['status'] == 'cancelled'
            survivor = submit('survives-other-cancellation')
            request('POST', api, f"/api/jobs/{first['id']}/cancel")
            assert wait(first['id'], {'cancelled'})['status'] == 'cancelled'
            control(delay=1)
            assert wait(survivor['id'], {'succeeded', 'failed'})['status'] == 'succeeded'
            assert request('GET', fake, '/metrics')['cancelled'] == [running['comfy_id']]
            return {'target': first['id'], 'queued_cancelled': queued['id'], 'survivor': survivor['id']}
        case('queued_and_targeted_running_cancellation', cancellation)

        def retry():
            control(fail_next=1, delay=1)
            failed = wait(submit('retry-with-same-seed')['id'], {'failed'})
            retried = request('POST', api, f"/api/jobs/{failed['id']}/retry", headers={'Idempotency-Key': 'retry-key'})
            assert retried['id'] != failed['id'] and retried['params']['seed'] == failed['params']['seed']
            assert wait(retried['id'], {'succeeded', 'failed'})['status'] == 'succeeded'
            return {'failed': failed['id'], 'retry': retried['id'], 'seed': retried['params']['seed']}
        case('failure_then_explicit_retry', retry)

        def restart():
            nonlocal service
            control(delay=5)
            job = wait(submit('resume-after-api-crash')['id'], {'running'})
            before = request('GET', fake, '/metrics')['submitted']
            service.kill()
            service.wait(timeout=10)
            service = start('backend.app:factory', api_port, 'api', environment)
            finished = wait(job['id'], {'succeeded', 'failed'})
            assert finished['status'] == 'succeeded'
            assert request('GET', fake, '/metrics')['submitted'] == before
            return {'job_id': job['id'], 'resubmitted': False}
        case('process_crash_recovers_known_prompt', restart)

        def outage():
            control(delay=5)
            job = wait(submit('temporary-upstream-outage')['id'], {'running'})
            control(offline=True)
            time.sleep(3)
            interrupted = request('GET', api, '/api/jobs/' + job['id'])
            assert interrupted['status'] == 'running'
            assert '重新连接' in interrupted['message']
            control(offline=False)
            assert wait(job['id'], {'succeeded', 'failed'})['status'] == 'succeeded'
            return {'job_id': job['id'], 'observed_message': interrupted['message']}
        case('upstream_outage_recovers_without_resubmit', outage)

        def missing():
            control(delay=20)
            job = wait(submit('lost-upstream-task')['id'], {'running'})
            control(clear=True)
            failed = wait(job['id'], {'failed'})
            assert '找不到' in failed['error']
            return {'job_id': job['id'], 'error': failed['error']}
        case('missing_upstream_task_fails_explicitly', missing)

        def uncertain_ack():
            nonlocal service
            control(delay=1, ack_delay=8)
            before = request('GET', fake, '/metrics')['submitted']
            job = submit('crash-between-submit-and-ack')
            for _ in range(100):
                if request('GET', fake, '/metrics')['submitted'] > before:
                    break
                time.sleep(.1)
            else:
                raise AssertionError('Upstream did not receive request')
            service.kill()
            service.wait(timeout=10)
            control(ack_delay=0)
            service = start('backend.app:factory', api_port, 'api', environment)
            failed = wait(job['id'], {'failed'})
            assert '无法确认' in failed['error']
            assert request('GET', fake, '/metrics')['submitted'] == before + 1
            return {'job_id': job['id'], 'resubmitted': False, 'error': failed['error']}
        case('uncertain_submission_not_automatically_duplicated', uncertain_ack)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        client.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
