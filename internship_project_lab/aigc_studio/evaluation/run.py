"""Resumable, isolated generation acceptance run. Waits for idle GPUs; never kills unrelated jobs."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from urllib.parse import urlsplit

import httpx

from backend.domain import JobInput, MODES, workflow
from backend.engine import Comfy
from backend.store import Store
from .support import DATA, SOURCE, Monitor, execution_times, freeze, nvidia, read, request, save


class Evaluation:
    def __init__(self, root, gpu='auto'):
        self.root, self.gpu = root, gpu
        self.protocol = read(SOURCE / 'evaluation/protocol.json')
        self.processes, self.workers = [], []
        self.monitor = None
        self.api = 'http://127.0.0.1:18810'
        self.client = httpx.Client(timeout=30, trust_env=False)
        self.root.mkdir(parents=True, exist_ok=True)
        freeze(root, self.protocol)

    def state(self, phase, **fields):
        save(self.root / 'state.json', {'phase': phase, 'updated': time.time(), **fields})
        print(json.dumps({'phase': phase, **fields}, ensure_ascii=False), flush=True)
        self.report()

    def report(self):
        from .report import render
        render(self.root)

    def spawn(self, command, log_name, env=None):
        with (self.root / log_name).open('a') as log:
            process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=log)
        self.processes.append(process)
        return process

    def add_worker(self, number):
        service_port = 18811 + number
        ident = f'eval-{number}'
        process = self.spawn([sys.executable, '-B', 'scripts/worker.py', '--gpu', self.gpu if number == 0 else 'auto', '--port', str(service_port), '--wait'], f'{ident}.log')
        address = f'http://127.0.0.1:{service_port}'
        self.state('waiting_for_gpu', requested_worker=ident, message='Existing compute jobs are left untouched')
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Worker exited: {ident}; inspect {ident}.log')
            marker = read(DATA / 'run' / f'comfy-{service_port}' / 'worker.json')
            if marker and marker['pid'] == process.pid:
                try:
                    request('GET', address, '/system_stats')
                    worker = {'id': ident, 'url': address, 'pid': process.pid, 'gpu': marker['gpu'], 'models': ['sdxl', 'wan22'], 'timeout_seconds': self.protocol['timeout_seconds']}
                    self.workers.append(worker)
                    save(self.root / 'workers.json', self.workers)
                    return worker
                except httpx.HTTPError:
                    pass
            time.sleep(10)

    def start_api(self):
        api_root = self.root / 'api'
        api_root.mkdir(exist_ok=True)
        save(api_root / 'settings.json', {'workers': self.workers})
        environment = {**os.environ, 'AIGC_DATA_ROOT': str(api_root)}
        process = self.spawn([sys.executable, '-B', '-m', 'uvicorn', 'backend.app:factory', '--factory', '--host', '127.0.0.1', '--port', str(urlsplit(self.api).port)], 'api.log', environment)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Evaluation API failed to start; inspect api.log')
            try:
                request('GET', self.api, '/api/health')
                self.api_process = process
                return
            except httpx.HTTPError:
                time.sleep(.2)
        raise TimeoutError('Evaluation API startup timed out')

    def clean_device(self, workers=None):
        devices = nvidia()
        for worker in workers or self.workers:
            other = [p['pid'] for p in devices[str(worker['gpu'])]['processes'] if p['pid'] != worker['pid']]
            if other:
                raise RuntimeError(f'GPU {worker["gpu"]} now contains other compute processes {other}; evaluation paused, not interrupting them')

    def upload(self, path):
        with Path(path).open('rb') as image:
            return request('POST', self.api, '/api/uploads', files={'file': (Path(path).name, image, 'image/png')})['id']

    def task(self, ident, params, phase, route='studio', checks=None, reference=None):
        record_path = self.root / 'tasks' / (ident + '.json')
        result = read(record_path, {})
        if result.get('finished'):
            return result
        self.clean_device()
        began = time.time()
        resumed = bool(result)
        result = {**result, 'id': ident, 'phase': phase, 'route': route, 'checks': checks or [], 'reference': reference, 'resumed': resumed,
                  'params': result.get('params', JobInput(**params).model_dump()), 'attempt_started': began}
        params = result['params']
        save(record_path, result)
        job = None
        worker = self.workers[0]
        try:
            if route == 'studio':
                job = request('POST', self.api, '/api/jobs', json=params, headers={'Idempotency-Key': self.root.name + ':' + ident})
                result['job_id'] = job['id']
            else:
                if result.get('dispatch_started') and not result.get('comfy_id'):
                    raise RuntimeError('Unknown direct submission outcome; inspect ComfyUI history before retrying')
                graph = workflow({'id': ident, 'model': MODES[params['mode']], 'params': params})
                save(self.root / 'workflows' / (ident + '.json'), graph)
                if not result.get('comfy_id'):
                    result['dispatch_started'] = time.time()
                    save(record_path, result)
                    result['comfy_id'] = request('POST', worker['url'], '/prompt', json={'prompt': graph, 'client_id': ident})['prompt_id']
            save(record_path, result)
            deadline = time.monotonic() + self.protocol['timeout_seconds']
            history = None
            while time.monotonic() < deadline:
                if route == 'studio':
                    job = request('GET', self.api, '/api/jobs/' + job['id'])
                    if job.get('worker_id'):
                        worker = next(w for w in self.workers if w['id'] == job['worker_id'])
                        result['worker_id'] = worker['id']
                    if job.get('comfy_id'):
                        result['comfy_id'] = job['comfy_id']
                    if job['status'] in {'failed', 'cancelled'}:
                        raise RuntimeError(job.get('error') or job['message'])
                    if job['status'] == 'succeeded':
                        worker = next(w for w in self.workers if w['id'] == job['worker_id'])
                        history = request('GET', worker['url'], '/history/' + job['comfy_id']).get(job['comfy_id'])
                        break
                else:
                    history = request('GET', worker['url'], '/history/' + result['comfy_id']).get(result['comfy_id'])
                    if history:
                        if history.get('status', {}).get('status_str') == 'error':
                            raise RuntimeError(json.dumps(history['status'].get('messages')))
                        if history.get('status', {}).get('completed'):
                            break
                time.sleep(.5)
            else:
                raise TimeoutError('Generation exceeded protocol timeout')
            result['observed_finished'] = time.time()
            if not history:
                raise ValueError('Missing true execution history')
            save(self.root / 'histories' / (ident + '.json'), history)
            result.update(execution_times(history))
            if result['execution_start'] is None or result['execution_end'] is None:
                raise ValueError('Execution timestamps missing')
            if 'sample' in result['cached_nodes']:
                raise ValueError('Sampler output was cached; not a valid generation measurement')
            suffix = '.png' if params['mode'] == 'text-to-image' else '.mp4'
            destination = self.root / 'media' / (ident + suffix)
            destination.parent.mkdir(exist_ok=True)
            if route == 'studio':
                result.update(accepted=job['created'], claimed=job['started'], output_ready=job['updated'], worker_id=job['worker_id'])
                response = self.client.get(self.api + '/api/assets/' + job['outputs'][0])
                response.raise_for_status()
                destination.write_bytes(response.content)
            else:
                store = Store(self.root / 'direct')
                direct_job = store.create(JobInput(**params))
                outputs = asyncio.run(Comfy(worker, store).download_outputs(direct_job, history))
                shutil.copyfile(store.asset_path(store.asset(outputs[0])), destination)
                result.update(accepted=result['dispatch_started'], claimed=result['dispatch_started'], output_ready=time.time(), worker_id=worker['id'])
            result['client_received'] = time.time()
            result['output'] = str(destination)
            inspection = subprocess.check_output([str(DATA / 'comfy-venv/bin/python'), '-B', str(SOURCE / 'evaluation/inspect_media.py'), str(destination), str(self.root / 'previews' / ident)], text=True, timeout=120)
            result['inspection'] = json.loads(inspection)
            observed = result['inspection']
            assert observed['width'] == params['width'] and observed['height'] == params['height'], 'Output dimensions mismatch'
            if params['mode'] != 'text-to-image':
                assert observed['frames'] == params['frames'], 'Output frame count mismatch'
                assert abs(observed['fps'] - params['fps']) < .01, 'Output FPS mismatch'
            result['success'] = True
        except Exception:
            result.update(success=False, error=traceback.format_exc())
            try:
                if route == 'studio' and job and job['status'] not in {'succeeded', 'failed', 'cancelled'}:
                    request('POST', self.api, '/api/jobs/' + job['id'] + '/cancel')
                elif route == 'direct' and result.get('comfy_id'):
                    request('POST', worker['url'], '/api/jobs/' + result['comfy_id'] + '/cancel')
            except Exception:
                pass
        result['finished'] = time.time()
        if job:
            save(self.root / 'jobs' / (ident + '.json'), job)
        result['resources'] = self.monitor.summarize(began, result.get('observed_finished', result['finished']), result.get('worker_id', worker['id']))
        save(record_path, result)
        print(json.dumps({'task': ident, 'success': result['success'], 'seconds': result['finished'] - began}, ensure_ascii=False), flush=True)
        return result

    def run(self, stage):
        self.add_worker(0)
        self.start_api()
        self.monitor = Monitor(self.root, self.workers)
        self.monitor.start()
        self.state('smoke')
        seed = self.protocol['seeds'][0]
        image = self.task('smoke-image', {'mode': 'text-to-image', 'prompt': 'A red ceramic teapot on a white table, photograph.', 'seed': seed, **self.protocol['smoke']}, 'smoke')
        if not image['success']:
            raise RuntimeError('Image smoke test failed; inspect task record')
        video = self.task('smoke-video', {'mode': 'text-to-video', 'prompt': 'A red ball slowly rolls across a wooden table. Fixed camera.', 'seed': seed, **self.protocol['smoke']}, 'smoke')
        if not video['success']:
            raise RuntimeError('Text-to-video smoke test failed; inspect task record')
        i2v = self.task('smoke-i2v', {'mode': 'image-to-video', 'prompt': 'The camera slowly moves closer to the teapot.', 'input_asset_id': self.upload(image['output']), 'seed': seed, **self.protocol['smoke']}, 'smoke', reference=image['output'])
        if not i2v['success']:
            raise RuntimeError('Image-to-video smoke test failed; inspect task record')
        if stage == 'smoke':
            self.state('smoke_complete')
            return
        self.state('dataset_images')
        for case in self.protocol['image_cases']:
            for seed in self.protocol['seeds']:
                self.task(f'dataset-{case["id"]}-{seed}', {'mode': 'text-to-image', 'prompt': case['prompt'], 'seed': seed, **self.protocol['image']}, 'dataset', checks=case['checks'])
                self.report()
        self.state('dataset_text_to_video')
        for case in self.protocol['video_cases']:
            for seed in self.protocol['seeds']:
                self.task(f'dataset-{case["id"]}-{seed}', {'mode': 'text-to-video', 'prompt': case['prompt'], 'seed': seed, **self.protocol['video']}, 'dataset', checks=case['checks'])
                self.report()
        self.state('dataset_image_to_video')
        for case in self.protocol['image_cases'][:10]:
            reference = read(self.root / 'tasks' / f'dataset-{case["id"]}-{self.protocol["seeds"][0]}.json')
            for seed in self.protocol['seeds']:
                ident = f'dataset-i2v-{case["id"]}-{seed}'
                if not reference or not reference['success']:
                    save(self.root / 'tasks' / (ident + '.json'), {'id': ident, 'phase': 'dataset', 'finished': time.time(), 'success': False, 'dependency_failed': True, 'params': {'mode': 'image-to-video'}, 'error': 'Preselected reference image failed'})
                    continue
                self.task(ident, {'mode': 'image-to-video', 'prompt': case['motion'], 'input_asset_id': self.upload(reference['output']), 'seed': seed, **self.protocol['video']}, 'dataset', checks=case['checks'] + [case['motion']], reference=reference['output'])
                self.report()
        self.state('paired_service_overhead')
        # Two passes alternate AB/BA by case and separate identical graphs. Classic
        # Comfy caching retains loaders but cannot reuse the previous sampler result.
        for pass_index in range(2):
            for index, case in enumerate(self.protocol['image_cases'][:8]):
                route = ['direct', 'studio'][(index + pass_index) % 2]
                self.task(f'paired-{case["id"]}-{route}', {'mode': 'text-to-image', 'prompt': case['prompt'], 'seed': 424242 + index, **self.protocol['image']}, 'paired', route=route)
                self.report()
        self.state('single_gpu_throughput')
        self.throughput('one_gpu')
        self.add_worker(1)
        self.api_process.terminate()
        self.api_process.wait(timeout=15)
        self.start_api()
        self.state('two_gpu_throughput')
        self.throughput('two_gpu')
        self.state('generation_complete', manual_review='Review materials prepared; no independent human ratings collected')

    def throughput(self, label):
        previous = read(self.root / (label + '.json'))
        if previous:
            return
        # An interrupted batch cannot be timed using already completed cached task
        # records. Preserve the old attempt and give the entire new batch a new ID.
        batch_id = label + '-' + uuid.uuid4().hex[:8]
        self.clean_device()
        # Warm every worker with a distinct prompt before the measured window.
        for index, worker in enumerate(self.workers):
            response = request('POST', worker['url'], '/prompt', json={'prompt': workflow({'id': 'warm-' + label + str(index), 'model': 'sdxl', 'params': JobInput(mode='text-to-image', prompt='A ceramic cup on a table.', seed=989000 + index, **self.protocol['image']).model_dump()})})
            until = time.monotonic() + self.protocol['timeout_seconds']
            while time.monotonic() < until:
                record = request('GET', worker['url'], '/history/' + response['prompt_id']).get(response['prompt_id'])
                if record:
                    if not record.get('status', {}).get('completed') or record.get('status', {}).get('status_str') == 'error':
                        raise RuntimeError('Warm-up failed')
                    break
                time.sleep(1)
            else:
                raise TimeoutError('Warm-up timed out')
        begin = time.time()
        def run_case(index):
            case = self.protocol['image_cases'][index]
            return self.task(f'{batch_id}-{case["id"]}', {'mode': 'text-to-image', 'prompt': case['prompt'], 'seed': 585000 + index, **self.protocol['image']}, label)
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(run_case, range(8)))
        end = time.time()
        successful = sum(bool(r['success']) for r in results)
        save(self.root / (label + '.json'), {'batch_id': batch_id, 'task_ids': [r['id'] for r in results], 'workers': len(self.workers), 'started': begin, 'ended': end, 'tasks': len(results), 'successful': successful, 'seconds': end-begin, 'images_per_hour': 3600 * successful / (end-begin), 'all_devices_clean': all(r['resources']['clean_device_observed'] for r in results), 'window_definition': 'client submission through full output decode of all tasks; includes validation overhead'})

    def close(self):
        if self.monitor:
            self.monitor.stop()
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='acceptance-v1')
    parser.add_argument('--gpu', default='auto')
    parser.add_argument('--stage', choices=['smoke', 'full', 'report', 'status'], default='full')
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('Invalid run id')
    root = DATA / 'evaluations' / args.run_id
    if args.stage in {'report', 'status'}:
        from .report import render
        render(root)
        print(json.dumps(read(root / 'state.json', {}), indent=2, ensure_ascii=False))
        print(str(root / 'report.md'))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluation = None
        try:
            from .report import render
            save(root / 'state.json', {'phase': 'initializing', 'updated': time.time()})
            render(root)
            evaluation = Evaluation(root, args.gpu)
            evaluation.run(args.stage)
        except Exception:
            save(root / 'state.json', {'phase': 'stopped_with_error', 'updated': time.time(), 'error': traceback.format_exc()})
            render(root)
            raise
        finally:
            if evaluation:
                evaluation.close()


if __name__ == '__main__':
    main()
