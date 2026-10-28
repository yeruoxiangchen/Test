import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.domain import JobInput, workflow
from backend.engine import Comfy
from backend.store import Store


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), '#879b70').save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        yield client


def test_api_idempotency_persistence_and_retry(client, tmp_path):
    payload = {'mode': 'text-to-image', 'prompt': '  a ceramic vase  '}
    headers = {'Idempotency-Key': 'first-submit'}
    first = client.post('/api/jobs', json=payload, headers=headers)
    assert first.status_code == 201
    job = first.json()
    assert job['params']['prompt'] == 'a ceramic vase'
    assert job['params']['seed'] >= 0
    assert client.post('/api/jobs', json=payload, headers=headers).json()['id'] == job['id']
    assert client.post('/api/jobs', json={**payload, 'prompt': 'different'}, headers=headers).status_code == 409
    assert Store(tmp_path).get(job['id'])['params']['seed'] == job['params']['seed']
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()['status'] == 'cancelled'
    retry = client.post(f"/api/jobs/{job['id']}/retry", headers={'Idempotency-Key': 'retry'}).json()
    assert retry['id'] != job['id']
    assert retry['retry_of'] == job['id']
    assert retry['params']['seed'] == job['params']['seed']
    assert client.post(f"/api/jobs/{job['id']}/retry", headers={'Idempotency-Key': 'retry'}).json()['id'] == retry['id']


@pytest.mark.parametrize('changes', [
    {'prompt': '   '}, {'mode': 'invalid'}, {'width': 1001}, {'height': 2048},
    {'mode': 'image-to-video'}, {'mode': 'text-to-video', 'frames': 50},
    {'input_asset_id': 'unwanted'}, {'guidance': 100}, {'seed': -2},
])
def test_invalid_requests_do_not_enter_queue(client, changes):
    assert client.post('/api/jobs', json={'mode': 'text-to-image', 'prompt': 'test', **changes}).status_code == 422
    assert client.get('/api/jobs').json()['items'] == []


def test_upload_validation_download_and_origin(client):
    assert client.post('/api/uploads', files={'file': ('x.png', b'not a png', 'image/png')}).status_code == 400
    upload = client.post('/api/uploads', files={'file': ('../../outside.png', png(), 'image/png')})
    assert upload.status_code == 201
    asset = upload.json()
    assert asset['width'] == 64
    image = client.get(asset['url'])
    assert image.status_code == 200 and image.headers['content-type'] == 'image/png'
    assert 'attachment' in client.get(asset['url'] + '?download=true').headers['content-disposition']
    assert client.get('/api/assets/nonexistent').status_code == 404
    body = {'mode': 'image-to-video', 'prompt': 'camera moves', 'input_asset_id': asset['id']}
    assert client.post('/api/jobs', json=body).status_code == 201
    assert client.post('/api/jobs', json={**body, 'input_asset_id': 'missing'}).status_code == 400
    assert client.post('/api/jobs', json=body, headers={'Origin': 'https://unrelated.example'}).status_code == 403


def test_atomic_claims_and_terminal_state(tmp_path):
    store = Store(tmp_path)
    job = store.create(JobInput(mode='text-to-image', prompt='only once'))
    with ThreadPoolExecutor(8) as pool:
        claimed = list(pool.map(lambda i: Store(tmp_path).claim(f'worker-{i}', ['sdxl']), range(8)))
    winners = [j for j in claimed if j]
    assert len(winners) == 1 and winners[0]['id'] == job['id']
    assert store.claim(winners[0]['worker_id'], ['sdxl']) is None
    store.cancel(job['id'])
    assert store.get(job['id'])['cancel_requested']
    store.update(job['id'], status='cancelled')
    store.update(job['id'], status='succeeded')
    assert store.get(job['id'])['status'] == 'cancelled'


def test_routes_only_to_available_model(tmp_path):
    store = Store(tmp_path)
    first = store.create(JobInput(mode='text-to-video', prompt='video'))
    second = store.create(JobInput(mode='text-to-image', prompt='image'))
    assert store.claim('image-only', ['sdxl'])['id'] == second['id']
    assert store.claim('offline', []) is None
    assert store.claim('video-only', ['wan22'])['id'] == first['id']


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['text-to-image', 'text-to-video', 'image-to-video'])
async def test_submit_resume_and_collect_outputs(tmp_path, monkeypatch, mode):
    store = Store(tmp_path)
    asset_id = None
    if mode == 'image-to-video':
        path = tmp_path / 'assets' / 'input.png'
        path.write_bytes(png())
        asset_id = store.add_asset(path, 'image/png', 'upload')['id']
    created = store.create(JobInput(mode=mode, prompt='integration fixture', input_asset_id=asset_id))
    job = store.claim('test-worker', [created['model']])
    content = png() if mode == 'text-to-image' else b'test-video-container'
    suffix = 'png' if mode == 'text-to-image' else 'mp4'
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == '/upload/image':
            return httpx.Response(200, json={'name': 'reference.png', 'subfolder': ''})
        if request.url.path == '/prompt':
            graph = json.loads(request.content)['prompt']
            assert graph['positive']['inputs']['text'] == 'integration fixture'
            if mode == 'image-to-video':
                assert graph['latent']['inputs']['start_image'] == ['image', 0]
                assert graph['image']['inputs']['image'] == 'reference.png'
            return httpx.Response(200, json={'prompt_id': 'comfy-123', 'node_errors': {}})
        if request.url.path == '/history/comfy-123':
            return httpx.Response(200, json={'comfy-123': {'status': {'completed': True}, 'outputs': {
                'preview': {'images': [{'filename': 'preview.png', 'type': 'temp'}]},
                'save': {'images': [{'filename': 'result.' + suffix, 'type': 'output', 'subfolder': 'aigc'}]},
            }}})
        if request.url.path == '/view':
            assert request.url.params['type'] == 'output'
            return httpx.Response(200, content=content)
        raise AssertionError(request.url)

    mock_http(monkeypatch, handler)
    worker = Comfy({'id': 'test-worker', 'url': 'http://fake'}, store)
    ident = await worker.submit(job)
    store.update(job['id'], status='running', comfy_id=ident)
    resumed = Store(tmp_path)
    await Comfy(worker.config, resumed).monitor(resumed.active('test-worker'))
    finished = resumed.get(job['id'])
    assert finished['status'] == 'succeeded'
    assert len(finished['outputs']) == 1
    asset = resumed.asset(finished['outputs'][0])
    assert resumed.asset_path(asset).read_bytes() == content
    assert asset['kind'] == ('image' if mode == 'text-to-image' else 'video')
    assert (('POST', '/upload/image') in requests) == (mode == 'image-to-video')


@pytest.mark.asyncio
async def test_cancellation_is_targeted_and_accepts_empty_response(tmp_path, monkeypatch):
    store = Store(tmp_path)
    job = store.create(JobInput(mode='text-to-image', prompt='cancel me'))
    store.claim('worker', ['sdxl'])
    store.update(job['id'], status='running', comfy_id='own-prompt')
    store.cancel(job['id'])
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.method == 'POST'
        return httpx.Response(200)

    mock_http(monkeypatch, handler)
    await Comfy({'id': 'worker', 'url': 'http://fake'}, store).monitor(store.get(job['id']))
    assert calls == ['/api/jobs/own-prompt/cancel']
    assert store.get(job['id'])['status'] == 'cancelled'


@pytest.mark.asyncio
async def test_missing_prompt_reports_failure(tmp_path, monkeypatch):
    store = Store(tmp_path)
    job = store.create(JobInput(mode='text-to-image', prompt='lost'))
    store.claim('worker', ['sdxl'])
    store.update(job['id'], status='running', comfy_id='gone')
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={}))

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    with pytest.raises(ValueError, match='找不到'):
        await Comfy({'id': 'worker', 'url': 'http://fake'}, store).monitor(store.get(job['id']))


def test_workflow_uses_request_parameters(tmp_path):
    spec = JobInput(mode='text-to-video', prompt='a bird', negative_prompt='text', width=704, height=1280, frames=33, seed=42, fps=16, steps=12)
    graph = workflow(Store(tmp_path).create(spec))
    assert graph['sample']['inputs']['seed'] == 42
    assert graph['sample']['inputs']['steps'] == 12
    assert graph['latent']['inputs']['width'] == 704
    assert graph['latent']['inputs']['length'] == 33
    assert graph['video']['inputs']['fps'] == 16
    assert graph['negative']['inputs']['text'] == 'text'
