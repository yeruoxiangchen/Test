"""Download resumable, revision-pinned weights into the project's data directory."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# The Xet client can recreate its output on retry. Use Hub's byte-range HTTP
# path so an interrupted file is appended from its existing length.
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

from huggingface_hub import HfApi, hf_hub_download
from requests import exceptions as requests_errors

SOURCE = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, ensure_ascii=False)
        temporary = Path(tmp.name)
    temporary.replace(path)


def write_state(root, status, **fields):
    save_json(root / 'models' / 'download-state.json', {'status': status, 'updated': time.time(), **fields})


def brief_error(exc):
    # Signed CDN URLs can be extremely long; keep persistent errors readable.
    return type(exc).__name__ + ': ' + re.sub(r'https?://\S+', '[download URL]', str(exc))[:500]


def is_transient(exc):
    for _ in range(8):
        if isinstance(exc, requests_errors.HTTPError) and exc.response is not None:
            return exc.response.status_code in {408, 429} or exc.response.status_code >= 500
        if isinstance(exc, (ConnectionError, TimeoutError, requests_errors.ConnectionError, requests_errors.Timeout,
                            requests_errors.ChunkedEncodingError, requests_errors.ContentDecodingError)):
            return True
        # Hub metadata failures can wrap a requests error in LocalEntryNotFoundError.
        exc = getattr(exc, '__cause__', None)
        if exc is None:
            return False
    return False


def retry_transfer(operation, root, key, attempts=8):
    for attempt in range(1, attempts + 1):
        write_state(root, 'downloading', key=key, attempt=attempt)
        try:
            return operation()
        except Exception as exc:
            if not is_transient(exc) or attempt == attempts:
                raise
            delay = min(300, 15 * 2 ** (attempt - 1))
            error = brief_error(exc)
            write_state(root, 'retrying', key=key, attempt=attempt, error=error, next_retry=time.time() + delay)
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S %Z")} Retry {attempt}/{attempts}: {error}; retry in {delay}s; cached data retained', flush=True)
            time.sleep(delay)


def prepare_http_transport(root, catalog):
    marker = root / 'models' / 'download-transport.json'
    if marker.exists():
        return
    preserved = []
    for model, files in catalog.items():
        for item in files:
            cache = root / 'models' / 'downloads' / model / '.cache' / 'huggingface' / 'download' / Path(item['file']).parent
            for partial in cache.glob('*.incomplete'):
                # Do not assume a Xet reconstruction is a valid contiguous HTTP
                # prefix. Preserve it separately before switching transports.
                backup = partial.with_name(partial.name + f'.xet-backup-{time.time_ns()}')
                partial.rename(backup)
                preserved.append(str(backup))
    save_json(marker, {'transport': 'http-range', 'preserved': preserved, 'updated': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['sdxl', 'wan22', 'all'], default='all')
    args = parser.parse_args()
    root = Path(os.environ['AIGC_DATA_ROOT'])
    (root / 'run').mkdir(parents=True, exist_ok=True)
    lock = (root / 'run' / 'download.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, 'A model downloader is already running for this data directory.\n')
    write_state(root, 'preparing')
    try:
        download_models(root, args.model)
    except Exception as exc:
        error = brief_error(exc)
        write_state(root, 'failed', error=error, retryable=is_transient(exc))
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S %Z")} Download failed: {error}', file=sys.stderr, flush=True)
        # The service retries network failures. Invalid configuration/auth/disk
        # errors stop with exit 2 instead of repeatedly making the same request.
        return 1 if is_transient(exc) else 2
    write_state(root, 'complete')
    return 0


def download_models(root, selected):
    manifest_path = root / 'models' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    plan_path = root / 'models' / 'download-plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    metadata_path = root / 'runtime' / 'model-download-metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    catalog = json.loads((SOURCE / 'config' / 'models.json').read_text())
    prepare_http_transport(root, catalog)
    api, revisions = HfApi(), {}
    for model, files in catalog.items():
        if selected not in ['all', model]:
            continue
        for item in files:
            name = Path(item['file']).name
            key = model + '/' + name
            old = manifest.get(key, {})
            repo = item['repo']
            revision = old.get('revision') or plan.get(key, {}).get('revision') or metadata.get(key, {}).get('revision') or revisions.get(repo)
            if not revision:
                revision = retry_transfer(lambda: api.model_info(repo).sha, root, key)
            revisions[repo] = revision
            plan[key] = {**item, 'revision': revision}
            save_json(plan_path, plan)
            print(f'Downloading {key} at {revision}', flush=True)
            print('Transport: HTTP Range; existing HTTP partial files will be resumed', flush=True)
            path = Path(retry_transfer(lambda: hf_hub_download(repo, item['file'], revision=revision, local_dir=root / 'models' / 'downloads' / model), root, key))
            folder = root / 'models' / item['folder']
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            if not target.exists():
                target.symlink_to(path)
            elif target.resolve() != path.resolve():
                raise RuntimeError(f'Refusing to replace an existing model: {target}')
            manifest[key] = {**item, 'revision': revision, 'path': str(path), 'bytes': path.stat().st_size}
            save_json(manifest_path, manifest)
            print(f'Ready: {target}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
