"""CPU-only fault-injection fixture. Not a generation backend or quality sample."""
import io
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image
import asyncio

app = FastAPI()
prompts = {}
controls = {'offline': False, 'delay': 2, 'fail_next': 0, 'ack_delay': 0}
metrics = {'submitted': 0, 'cancelled': []}


def check():
    if controls['offline']:
        raise HTTPException(503, 'Injected upstream outage')


@app.post('/control')
async def control(request: Request):
    body = await request.json()
    if body.pop('clear', False):
        prompts.clear()
    controls.update(body)
    return controls


@app.get('/metrics')
def stats():
    return metrics


@app.get('/object_info')
def info():
    check()
    return {'CheckpointLoaderSimple': {'input': {'required': {'ckpt_name': [['sd_xl_base_1.0.safetensors']]}}}, 'SaveImage': {}}


@app.post('/prompt')
async def submit(request: Request):
    check()
    await request.json()
    ident = str(uuid.uuid4())
    now = time.time()
    failing = controls['fail_next'] > 0
    controls['fail_next'] = max(0, controls['fail_next'] - 1)
    prompts[ident] = {'start': now, 'end': now + controls['delay'], 'fail': failing}
    metrics['submitted'] += 1
    await asyncio.sleep(controls['ack_delay'])
    return {'prompt_id': ident, 'node_errors': {}}


@app.get('/queue')
def queue():
    check()
    active = [[i, ident] for i, (ident, p) in enumerate(prompts.items()) if p['end'] > time.time() and not p.get('cancelled')]
    return {'queue_running': active, 'queue_pending': []}


@app.get('/history/{ident}')
def history(ident: str):
    check()
    p = prompts.get(ident)
    if not p or p['end'] > time.time():
        return {}
    failed = p['fail'] or p.get('cancelled', False)
    return {ident: {'status': {'completed': not failed, 'status_str': 'error' if failed else 'success', 'messages': [
        ['execution_start', {'timestamp': int(p['start'] * 1000)}],
        ['execution_error' if failed else 'execution_success', {'timestamp': int(p['end'] * 1000), 'exception_message': 'Injected failure' if failed else ''}],
    ]}, 'outputs': {} if failed else {'save': {'images': [{'filename': ident + '.png', 'subfolder': '', 'type': 'output'}]}}}}


@app.post('/api/jobs/{ident}/cancel')
def cancel(ident: str):
    check()
    if ident in prompts:
        prompts[ident].update(cancelled=True, end=time.time())
    metrics['cancelled'].append(ident)
    return {'cancelled': True}


@app.get('/view')
def view():
    check()
    data = io.BytesIO()
    Image.new('RGB', (512, 512), '#b3c893').save(data, format='PNG')
    return Response(data.getvalue(), media_type='image/png')
