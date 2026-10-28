from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from .domain import CATALOG, ROOT, JobInput
from .engine import Comfy
from .store import Conflict, Store

Image.MAX_IMAGE_PIXELS = 25_000_000


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        return [{'index': int(a), 'name': b.strip(), 'total_mib': int(c), 'free_mib': int(d), 'utilization': int(e)} for a, b, c, d, e in (line.split(',') for line in result.stdout.strip().splitlines())]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def create_app(root: Path = ROOT, start_workers=True):
    store = Store(root)
    config_path = root / 'settings.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {'workers': [{'id': 'local-1', 'url': 'http://127.0.0.1:18791', 'models': ['sdxl', 'wan22']}]}
    configs = config.get('workers', [])
    if len({w['id'] for w in configs}) != len(configs) or len({w['url'].rstrip('/') for w in configs}) != len(configs):
        raise ValueError('Worker id 和推理地址不能重复')
    workers = [Comfy(w, store) for w in configs]

    @asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(w.loop()) for w in workers] if start_workers else []
        yield
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title='帧间 Studio', lifespan=lifespan)
    app.state.store, app.state.workers = store, workers

    @app.middleware('http')
    async def same_origin(request: Request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if origin:
                from urllib.parse import urlsplit
                if urlsplit(origin).netloc != request.headers.get('host'):
                    return JSONResponse({'detail': '不允许跨站修改任务'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/health')
    def health():
        return {'ok': True, 'version': '0.1.0'}

    @app.get('/api/system')
    async def system():
        states = await asyncio.gather(*(w.readiness() for w in workers))
        ready = {m for state in states for m in state['ready_models']}
        return {'models': [{**m, 'ready': m['id'] in ready} for m in CATALOG], 'workers': states,
                'gpus': await asyncio.to_thread(gpu_snapshot), 'timestamp': time.time()}

    @app.get('/api/jobs')
    def jobs():
        return {'items': store.list_jobs()}

    @app.post('/api/jobs', status_code=201)
    def submit(spec: JobInput, idempotency_key: str | None = Header(default=None)):
        if idempotency_key and len(idempotency_key) > 100:
            raise HTTPException(400, '请求标识过长')
        if spec.input_asset_id:
            asset = store.asset(spec.input_asset_id)
            if not asset or asset['kind'] != 'upload' or not store.asset_path(asset).exists():
                raise HTTPException(400, '参考图片不存在，请重新上传')
        try:
            return store.create(spec, idempotency_key)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/jobs/{ident}')
    def get_job(ident: str):
        job = store.get(ident)
        if not job:
            raise HTTPException(404, '任务不存在')
        return job

    @app.post('/api/jobs/{ident}/cancel')
    def cancel(ident: str):
        job = store.cancel(ident)
        if not job:
            raise HTTPException(404, '任务不存在')
        return job

    @app.post('/api/jobs/{ident}/retry', status_code=201)
    def retry(ident: str, idempotency_key: str | None = Header(default=None)):
        old = get_job(ident)
        if old['status'] not in {'failed', 'cancelled', 'succeeded'}:
            raise HTTPException(409, '请等待任务结束后再重新生成')
        if idempotency_key and len(idempotency_key) > 100:
            raise HTTPException(400, '请求标识过长')
        try:
            return store.create(JobInput(**old['params']), idempotency_key, retry_of=ident)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/api/uploads', status_code=201)
    async def upload(file: UploadFile = File(...)):
        data = await file.read(20 * 1024 * 1024 + 1)
        await file.close()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, '图片不能超过 20MB')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as original:
                    if original.format not in {'PNG', 'JPEG', 'WEBP'}:
                        raise ValueError('请上传 PNG、JPEG 或 WebP 图片')
                    from PIL import ImageOps
                    image = ImageOps.exif_transpose(original).convert('RGB')
                    image.load()
            path = store.root / 'assets' / (uuid.uuid4().hex + '.png')
            image.save(path)
            asset = store.add_asset(path, 'image/png', 'upload')
            return {'id': asset['id'], 'url': '/api/assets/' + asset['id'], 'width': image.width, 'height': image.height}
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise HTTPException(400, '无法读取图片，请使用有效的 PNG、JPEG 或 WebP 文件') from exc

    @app.get('/api/assets/{ident}')
    def asset(ident: str, download: bool = False):
        item = store.asset(ident)
        if not item:
            raise HTTPException(404, '作品不存在')
        path = store.asset_path(item)
        if not path.is_file():
            raise HTTPException(404, '作品文件不存在')
        return FileResponse(path, media_type=item['mime'], filename=path.name if download else None)

    @app.get('/{path:path}')
    def frontend(path: str):
        if path.startswith('api/'):
            raise HTTPException(404, '接口不存在')
        web = Path(os.environ.get('AIGC_WEB_ROOT', str(root / 'web-dist'))).resolve()
        target = (web / path).resolve()
        if not target.is_relative_to(web):
            raise HTTPException(404)
        if target.is_file():
            return FileResponse(target)
        if (web / 'index.html').is_file():
            return FileResponse(web / 'index.html')
        return JSONResponse({'detail': '前端尚未构建，请执行 scripts/run.sh build'}, status_code=503)

    return app


def factory():
    return create_app()
