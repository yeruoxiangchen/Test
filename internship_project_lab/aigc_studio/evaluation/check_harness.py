"""CPU integration check for measurement plumbing; never a model benchmark."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import httpx

from .faults import port
from .run import Evaluation
from .support import DATA, SOURCE, read, request, save


def main():
    root = DATA / 'checks' / ('evaluation-harness-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    evaluation = object.__new__(Evaluation)
    evaluation.root = root
    evaluation.protocol = read(SOURCE / 'evaluation/protocol.json')
    evaluation.processes, evaluation.workers = [], []
    evaluation.api = f'http://127.0.0.1:{port()}'
    evaluation.client = httpx.Client(timeout=30, trust_env=False)
    evaluation.clean_device = lambda: None  # fixture only, never used in real evaluation
    evaluation.monitor = SimpleNamespace(stop=lambda: None, summarize=lambda *args: {
        'samples': 0, 'clean_device_observed': False, 'note': 'CPU fixture, performance ineligible'})
    upstream_port = port()
    upstream = f'http://127.0.0.1:{upstream_port}'
    try:
        process = evaluation.spawn([sys.executable, '-B', '-m', 'uvicorn', 'evaluation.fake_comfy:app',
            '--host', '127.0.0.1', '--port', str(upstream_port)], 'fake.log')
        for _ in range(100):
            try:
                request('GET', upstream, '/metrics')
                break
            except httpx.HTTPError:
                if process.poll() is not None:
                    raise RuntimeError('CPU fixture exited')
                time.sleep(.1)
        else:
            raise TimeoutError('CPU fixture startup failed')
        evaluation.workers = [{'id': 'eval-0', 'url': upstream, 'models': ['sdxl'], 'timeout_seconds': 30}]
        evaluation.start_api()
        params = {'mode': 'text-to-image', 'prompt': 'CPU fixture; not generated content', 'seed': 42, 'width': 512, 'height': 512, 'steps': 6}
        results = [evaluation.task('fixture-' + route, params, 'fixture', route=route) for route in ['studio', 'direct']]
        assert all(r['success'] for r in results), results
        assert all(r['inspection']['decoded'] and r['execution_end'] > r['execution_start'] for r in results)
        request('POST', upstream, '/control', json={'fail_next': 1})
        failed = evaluation.task('fixture-failure', params, 'fixture')
        assert not failed['success'] and 'Injected failure' in failed['error']
        media_path = read(DATA / 'checks/comfy-report.json')['output']
        inspection = json.loads(subprocess.check_output([str(DATA / 'comfy-venv/bin/python'), '-B',
            str(SOURCE / 'evaluation/inspect_media.py'), media_path, str(root / 'video-preview')], text=True))
        assert (inspection['width'], inspection['height'], inspection['frames'], inspection['fps']) == (64, 64, 8, 16)
        evaluation.report()
        evidence = {'kind': 'CPU fixture only', 'passed': True, 'direct_and_api_records': [r['id'] for r in results],
                    'failure_preserved': True, 'full_video_decode': inspection, 'output': str(root)}
        save(DATA / 'checks/evaluation-harness.json', evidence)
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
    finally:
        evaluation.close()


if __name__ == '__main__':
    main()
