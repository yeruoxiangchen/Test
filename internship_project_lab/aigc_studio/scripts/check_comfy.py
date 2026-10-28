"""Check real ComfyUI node schemas and encode a CPU test-pattern MP4, without model inference."""
import argparse
import asyncio
import json
import uuid

import httpx

from backend.domain import ROOT, MODES, JobInput, workflow
from backend.engine import Comfy
from backend.store import Store


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18801')
    args = parser.parse_args()
    root = ROOT / 'checks' / ('comfy-' + uuid.uuid4().hex)
    store = Store(root)
    worker = Comfy({'id': 'cpu-check', 'url': args.url}, store)
    stats = await worker.request('GET', '/system_stats')
    assert all(d['type'] == 'cpu' for d in stats['devices']), 'Use a CPU-only diagnostic worker'
    info = await worker.request('GET', '/object_info')
    (root / 'object_info.json').write_text(json.dumps(info, ensure_ascii=False))
    file_fields = {'ckpt_name', 'unet_name', 'clip_name', 'vae_name', 'image'}
    for mode in MODES:
        spec = JobInput(mode=mode, prompt='schema check', input_asset_id='fixture' if mode == 'image-to-video' else None)
        job = store.create(spec)
        graph = workflow(job, 'fixture.png')
        for ident, node in graph.items():
            schema = info[node['class_type']]
            required = schema['input'].get('required', {})
            optional = schema['input'].get('optional', {})
            assert set(required) <= node['inputs'].keys(), (mode, ident, 'missing', set(required) - node['inputs'].keys())
            for name, value in node['inputs'].items():
                assert name in required or name in optional, (mode, ident, 'unknown input', name)
                field = required.get(name) or optional.get(name)
                if isinstance(field[0], list) and isinstance(value, str) and name not in file_fields:
                    assert value in field[0], (mode, ident, name, value, field)
                if isinstance(value, list):
                    assert value[0] in graph, (mode, ident, 'broken link', value)
        print(f'Node schema verified: {mode}', flush=True)
    graph = {
        'pattern': {'class_type': 'EmptyImage', 'inputs': {'width': 64, 'height': 64, 'batch_size': 8, 'color': 0xD7ED93}},
        'video': {'class_type': 'CreateVideo', 'inputs': {'images': ['pattern', 0], 'fps': 16.0}},
        'save': workflow(store.create(JobInput(mode='text-to-video', prompt='CPU encoding fixture')))['save'],
    }
    graph['save']['inputs']['filename_prefix'] = 'checks/cpu-pattern'
    response = await worker.request('POST', '/prompt', json={'prompt': graph})
    ident = response['prompt_id']
    record = None
    for _ in range(30):
        record = (await worker.request('GET', '/history/' + ident)).get(ident)
        if record:
            break
        await asyncio.sleep(1)
    assert record and record['status']['completed'], record
    test_job = store.create(JobInput(mode='text-to-video', prompt='CPU encoding fixture'))
    outputs = await worker.download_outputs(test_job, record)
    path = store.asset_path(store.asset(outputs[0]))
    with path.open('rb') as video:
        assert video.read(12)[4:8] == b'ftyp'
    report = {'schemas': list(MODES), 'cpu_video_encoding': True, 'output': str(path), 'gpu_generation_tested': False}
    (ROOT / 'checks' / 'comfy-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
