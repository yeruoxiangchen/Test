"""Freeze the 16 selected RGB/camera archive members for each training object.

This reads source-cache view IDs through mmap. It does not re-read all DINO
feature tensors and does not certify unused archive members. No GPU is used.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from ...paired_dataset import sha256_file, canonical_json_sha256
from ...prepare_official_experiment import read_selected_render_members

PACKAGE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PACKAGE / 'outputs/round1_20260909/data_v2/train2000.json'


def atomic(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(item, indent=2) + '\n')
    os.replace(tmp, path)


def check(row):
    source = row['source']
    cache = torch.load(source['cache'], map_location='cpu', mmap=True, weights_only=False)
    view_ids = [int(v) for v in cache['view_ids'].tolist()]
    del cache
    if len(view_ids) != 8 or len(set(view_ids)) != 8:
        raise ValueError('expected eight unique source views')
    names = [f"{source['uid']}/{view:03d}{suffix}" for view in view_ids
             for suffix in ('.rgba.webp', '.json')]
    members = read_selected_render_members(source['render'], names)
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}
    return {'uid': row['uid'], 'object_uid': row['object_uid'], 'source_cache': source['cache'],
        'source_cache_sha256': source['cache_sha256'], 'render_path': source['render'],
        'view_ids': view_ids, 'selected_render_member_sha256': hashes,
        'selected_member_bytes': sum(len(v) for v in members.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.output_dir).resolve()
    source = Path(args.manifest).resolve(strict=True)
    payload = json.loads(source.read_text())
    if payload['sample_count'] != 2000 or payload['object_count'] != 2000:
        raise ValueError('only the fixed clean training 2K is accepted')
    expected = {'manifest': {'path': str(source), 'sha256': sha256_file(source)},
        'code': {str(Path(__file__).resolve()): sha256_file(__file__),
                 str(PACKAGE / 'prepare_official_experiment.py'):
                     sha256_file(PACKAGE / 'prepare_official_experiment.py')}}
    final = root / 'render_source_audit.json'
    if final.exists():
        saved = json.loads(final.read_text())
        if saved['binding'] != expected or not saved['passed'] or len(saved['rows']) != 2000:
            raise ValueError('existing selected-source audit binding differs')
        print(json.dumps({'state': 'complete', 'resumed': True, 'count': len(saved['rows'])}), flush=True)
        return
    started, rows = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(check, payload['samples']), 1):
            rows.append(row)
            if i % 32 == 0 or i == len(payload['samples']):
                status = {'state': 'auditing_selected_render_members', 'completed': i,
                    'total': len(payload['samples']), 'elapsed_seconds': time.monotonic() - started,
                    'utc': datetime.now(timezone.utc).isoformat()}
                atomic(root / 'render_source_audit_progress.json', status)
                print(json.dumps(status), flush=True)
    result = {'passed': True, 'binding': expected, 'count': len(rows), 'rows': rows,
        'scope': 'selected eight RGBA and eight camera members per object; unused archive members not certified',
        'elapsed_seconds': time.monotonic() - started,
        'selected_bytes': sum(r['selected_member_bytes'] for r in rows),
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    result['identity_hash'] = canonical_json_sha256(result)
    atomic(final, result)
    print(json.dumps({'state': 'complete', 'count': len(rows), 'identity_hash': result['identity_hash']}), flush=True)


if __name__ == '__main__':
    main()
