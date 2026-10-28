from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
import uuid
from pathlib import Path

import httpx

from .domain import workflow
from .store import Store

log = logging.getLogger('studio.engine')


class Comfy:
    def __init__(self, config: dict, store: Store):
        self.config, self.store = config, store
        self.id = config['id']
        self.base = config['url'].rstrip('/')
        self.cache_time = 0.0
        self.snapshot = {'id': self.id, 'online': False, 'ready_models': [], 'message': '等待本地推理服务'}

    async def request(self, method, path, **kwargs):
        async with httpx.AsyncClient(base_url=self.base, timeout=30, trust_env=False) as client:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}

    async def readiness(self, force=False):
        if not force and time.monotonic() - self.cache_time < 10:
            return self.snapshot
        ready, missing = [], {}
        try:
            info = await self.request('GET', '/object_info')
            checks = {
                'sdxl': [('CheckpointLoaderSimple', 'ckpt_name', 'sd_xl_base_1.0.safetensors'), ('SaveImage', None, None)],
                'wan22': [('UNETLoader', 'unet_name', 'wan2.2_ti2v_5B_fp16.safetensors'),
                          ('CLIPLoader', 'clip_name', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'),
                          ('VAELoader', 'vae_name', 'wan2.2_vae.safetensors'),
                          ('Wan22ImageToVideoLatent', None, None), ('CreateVideo', None, None), ('SaveVideo', None, None)],
            }
            for model in self.config.get('models', ['sdxl', 'wan22']):
                absent = []
                for node, field, filename in checks[model]:
                    if node not in info:
                        absent.append(node)
                    elif field:
                        options = info[node].get('input', {}).get('required', {}).get(field, [[]])[0]
                        if filename not in options:
                            absent.append(filename)
                if absent:
                    missing[model] = absent
                else:
                    ready.append(model)
            self.snapshot = {'id': self.id, 'online': True, 'ready_models': ready, 'missing': missing,
                             'message': '生成服务已连接' if ready else '生成服务已连接，等待模型文件'}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            self.snapshot = {'id': self.id, 'online': False, 'ready_models': [], 'message': '等待本地推理服务', 'detail': type(exc).__name__}
        self.cache_time = time.monotonic()
        return self.snapshot

    async def submit(self, job):
        uploaded = None
        if job['params']['input_asset_id']:
            asset = self.store.asset(job['params']['input_asset_id'])
            if not asset:
                raise ValueError('参考图片不存在')
            with self.store.asset_path(asset).open('rb') as image:
                result = await self.request('POST', '/upload/image', files={'image': (asset['id'] + '.png', image, 'image/png')}, data={'overwrite': 'false'})
            uploaded = (result.get('subfolder', '').rstrip('/') + '/' + result['name']).lstrip('/')
        result = await self.request('POST', '/prompt', json={'prompt': workflow(job, uploaded), 'client_id': 'aigc-' + job['id']})
        if not result.get('prompt_id') or result.get('node_errors'):
            raise ValueError('推理工作流校验失败：' + json.dumps(result.get('node_errors') or result.get('error'), ensure_ascii=False)[:1000])
        return result['prompt_id']

    async def cancel(self, prompt_id):
        # The pinned ComfyUI revision checks the running prompt atomically.
        # Do not fall back to its global /interrupt endpoint.
        await self.request('POST', '/api/jobs/' + prompt_id + '/cancel')

    async def download_outputs(self, job, history):
        results = []
        for value in history.get('outputs', {}).values():
            for key in ['images', 'videos', 'gifs']:
                for item in value.get(key, []):
                    suffix = Path(item.get('filename', '')).suffix.lower()
                    if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.webm', '.gif'} or item.get('type') != 'output':
                        continue
                    ident = uuid.uuid4().hex
                    destination = self.store.root / 'assets' / (ident + suffix)
                    temporary = destination.with_suffix(suffix + '.part')
                    size = 0
                    try:
                        async with httpx.AsyncClient(base_url=self.base, timeout=120, trust_env=False) as client:
                            async with client.stream('GET', '/view', params={k: item.get(k, '') for k in ['filename', 'subfolder', 'type']}) as response:
                                response.raise_for_status()
                                with temporary.open('wb') as target:
                                    async for chunk in response.aiter_bytes(1024 * 1024):
                                        size += len(chunk)
                                        if size > 512 * 1024 * 1024:
                                            raise ValueError('单个输出文件超过 512MB 限制')
                                        target.write(chunk)
                        if not size:
                            raise ValueError('推理服务返回了空文件')
                        temporary.replace(destination)
                        kind = 'video' if suffix in {'.mp4', '.webm'} else 'image'
                        results.append(self.store.add_asset(destination, mimetypes.guess_type(destination)[0] or 'application/octet-stream', kind, job['id'])['id'])
                    finally:
                        temporary.unlink(missing_ok=True)
        if not results:
            raise ValueError('推理任务结束但没有可下载的图片或视频')
        return results

    async def monitor(self, job):
        elapsed = max(0, time.time() - (job['started'] or job['created']))
        deadline = time.monotonic() + max(0, self.config.get('timeout_seconds', 7200) - elapsed)
        missing_count = 0
        while time.monotonic() < deadline:
            current = self.store.get(job['id'])
            try:
                if current['cancel_requested']:
                    await self.cancel(job['comfy_id'])
                    self.store.update(job['id'], status='cancelled', message='已取消')
                    return
                data = await self.request('GET', '/history/' + job['comfy_id'])
                record = data.get(job['comfy_id'])
                if record:
                    status = record.get('status', {})
                    if status.get('status_str') == 'error':
                        raise ValueError('模型执行失败：' + json.dumps(status.get('messages', []), ensure_ascii=False)[-1800:])
                    if status.get('completed') or record.get('outputs'):
                        self.store.update(job['id'], message='正在保存作品')
                        outputs = await self.download_outputs(job, record)
                        if self.store.get(job['id'])['cancel_requested']:
                            self.store.update(job['id'], status='cancelled', message='已取消')
                        else:
                            self.store.update(job['id'], status='succeeded', message='生成完成', outputs=outputs)
                        return
                queue = await self.request('GET', '/queue')
                running = [item[1] for item in queue.get('queue_running', [])]
                pending = [item[1] for item in queue.get('queue_pending', [])]
                if job['comfy_id'] in running:
                    missing_count = 0
                    self.store.update(job['id'], message='模型正在生成，请稍候')
                elif job['comfy_id'] in pending:
                    missing_count = 0
                    self.store.update(job['id'], message='已提交，等待推理资源')
                else:
                    missing_count += 1
                    if missing_count >= 5:
                        raise ValueError('推理服务中已找不到此任务，可从原参数重试')
            except httpx.HTTPError:
                self.store.update(job['id'], message='推理连接暂时中断，正在重新连接')
            await asyncio.sleep(2)
        try:
            await self.cancel(job['comfy_id'])
        except httpx.HTTPError:
            log.warning('Unable to cancel timed-out prompt %s', job['comfy_id'])
        raise TimeoutError('任务超过运行时限；请检查推理服务后再重试')

    async def loop(self):
        while True:
            job = None
            try:
                job = self.store.active(self.id)
                if job:
                    if not job['comfy_id']:
                        raise ValueError('服务在提交期间重启，无法确认提交结果；请检查后手动重试')
                else:
                    state = await self.readiness()
                    job = self.store.claim(self.id, state['ready_models'])
                    if job:
                        if self.store.get(job['id'])['cancel_requested']:
                            self.store.update(job['id'], status='cancelled', message='已取消')
                            continue
                        ident = await self.submit(job)
                        self.store.update(job['id'], status='running', comfy_id=ident, message='已提交推理任务')
                        job = self.store.get(job['id'])
                if job:
                    await self.monitor(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception('Worker %s failed', self.id)
                if job:
                    self.store.update(job['id'], status='failed', error=str(exc)[:2000], message='生成失败，可查看详情并重试')
            await asyncio.sleep(2)

