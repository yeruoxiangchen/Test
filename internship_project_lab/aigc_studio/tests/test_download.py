import json

import pytest
import requests
from huggingface_hub.errors import LocalEntryNotFoundError

from scripts import prepare_models as downloader


def test_network_retry_preserves_partial_and_reports_backoff(tmp_path, monkeypatch):
    partial = tmp_path / 'weights.incomplete'
    partial.write_bytes(b'cached-prefix')
    calls, states = [], []

    def transfer():
        calls.append(1)
        assert partial.read_bytes() == b'cached-prefix'
        if len(calls) < 3:
            raise ConnectionError('network interrupted at https://example.test/signed?secret=value')
        return 'done'

    def sleep(delay):
        state = json.loads((tmp_path / 'models' / 'download-state.json').read_text())
        assert state['status'] == 'retrying'
        assert 'secret=value' not in state['error']
        states.append(delay)

    monkeypatch.setattr(downloader.time, 'sleep', sleep)
    assert downloader.retry_transfer(transfer, tmp_path, 'sdxl/file') == 'done'
    assert len(calls) == 3 and states == [15, 30]


@pytest.mark.parametrize('code,transient', [(401, False), (403, False), (404, False), (429, True), (503, True)])
def test_http_retry_policy(code, transient):
    response = requests.Response()
    response.status_code = code
    assert downloader.is_transient(requests.HTTPError(response=response)) is transient


def test_disk_failure_does_not_retry(tmp_path, monkeypatch):
    def transfer():
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: pytest.fail('Disk errors must not retry'))
    with pytest.raises(OSError):
        downloader.retry_transfer(transfer, tmp_path, 'sdxl/file')


def test_wrapped_metadata_timeout_is_retryable():
    wrapped = LocalEntryNotFoundError('Cannot locate file after metadata request failed')
    wrapped.__cause__ = requests.Timeout('metadata timed out')
    assert downloader.is_transient(wrapped)


def test_truncated_network_response_is_retryable():
    assert downloader.is_transient(requests.exceptions.ChunkedEncodingError('response ended early'))


def test_resume_uses_original_revision_and_registers_file(tmp_path, monkeypatch):
    metadata = tmp_path / 'runtime' / 'model-download-metadata.json'
    metadata.parent.mkdir()
    key = 'sdxl/sd_xl_base_1.0.safetensors'
    metadata.write_text(json.dumps({key: {'revision': 'pinned-revision'}}))
    result = tmp_path / 'models' / 'downloads' / 'sdxl' / 'sd_xl_base_1.0.safetensors'
    result.parent.mkdir(parents=True)
    result.write_bytes(b'fixture')

    def download(repo, filename, revision, local_dir):
        assert revision == 'pinned-revision'
        plan = json.loads((tmp_path / 'models' / 'download-plan.json').read_text())
        assert plan[key]['revision'] == revision
        return str(result)

    monkeypatch.setattr(downloader, 'hf_hub_download', download)
    downloader.download_models(tmp_path, 'sdxl')
    manifest = json.loads((tmp_path / 'models' / 'manifest.json').read_text())
    assert manifest[key]['revision'] == 'pinned-revision'
    assert (tmp_path / 'models' / 'checkpoints' / result.name).resolve() == result


def test_transport_change_preserves_xet_files_and_does_not_move_http_resume(tmp_path):
    cache = tmp_path / 'models' / 'downloads' / 'sdxl' / '.cache' / 'huggingface' / 'download'
    cache.mkdir(parents=True)
    partial = cache / 'model.hash.incomplete'
    partial.write_bytes(b'xet-reconstruction')
    catalog = {'sdxl': [{'file': 'model.safetensors'}]}
    downloader.prepare_http_transport(tmp_path, catalog)
    marker = json.loads((tmp_path / 'models' / 'download-transport.json').read_text())
    from pathlib import Path
    assert Path(marker['preserved'][0]).read_bytes() == b'xet-reconstruction'
    partial.write_bytes(b'http-prefix')
    downloader.prepare_http_transport(tmp_path, catalog)
    assert partial.read_bytes() == b'http-prefix'


def test_real_hub_http_download_resumes_range(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from huggingface_hub.file_download import http_get

    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')

    data = b'example-model-bytes-' * 1024
    offset, ranges = 137, []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ranges.append(self.headers.get('Range'))
            self.send_response(206)
            self.send_header('Content-Length', str(len(data) - offset))
            self.send_header('Content-Range', f'bytes {offset}-{len(data) - 1}/{len(data)}')
            self.end_headers()
            self.wfile.write(data[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / 'model.incomplete'
    path.write_bytes(data[:offset])
    try:
        with path.open('ab') as file:
            http_get(f'http://127.0.0.1:{server.server_port}/weights', file, resume_size=offset, expected_size=len(data))
        assert ranges == [f'bytes={offset}-']
        assert path.read_bytes() == data
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
