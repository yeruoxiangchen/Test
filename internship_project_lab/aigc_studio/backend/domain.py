from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.environ.get('AIGC_DATA_ROOT', '/data/zjr/aigc_studio')).resolve()
MODES = {'text-to-image': 'sdxl', 'text-to-video': 'wan22', 'image-to-video': 'wan22'}
TERMINAL = {'succeeded', 'failed', 'cancelled'}
CATALOG = [
    {'id': 'sdxl', 'name': 'SDXL 1.0', 'type': 'image', 'description': '文字生成图片 · 多种画幅',
     'modes': ['text-to-image'], 'default_steps': 25},
    {'id': 'wan22', 'name': 'Wan 2.2 · 5B', 'type': 'video', 'description': '文字或图片生成视频',
     'modes': ['text-to-video', 'image-to-video'], 'default_steps': 20},
]


class JobInput(BaseModel):
    mode: Literal['text-to-image', 'text-to-video', 'image-to-video']
    prompt: str = Field(min_length=1, max_length=6000)
    negative_prompt: str = Field(default='', max_length=3000)
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=25, ge=1, le=60)
    seed: int = Field(default=-1, ge=-1, le=2**53 - 1)
    guidance: float = Field(default=6.0, ge=1, le=15)
    frames: int = Field(default=49, ge=17, le=121)
    fps: int = Field(default=16, ge=8, le=30)
    input_asset_id: str | None = None

    @model_validator(mode='after')
    def check_shape(self):
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError('请填写提示词')
        if self.width % 32 or self.height % 32:
            raise ValueError('宽高必须为 32 的倍数')
        if self.width * self.height > 1536 * 1024:
            raise ValueError('画面面积超过当前服务上限')
        if self.mode != 'text-to-image' and (self.frames - 1) % 4:
            raise ValueError('视频帧数必须为 4n+1，例如 33、49、81')
        if self.mode == 'image-to-video' and not self.input_asset_id:
            raise ValueError('图生视频需要上传参考图片')
        if self.mode != 'image-to-video' and self.input_asset_id:
            raise ValueError('当前模式不使用参考图片')
        return self


def workflow(job: dict, uploaded_name: str | None = None) -> dict:
    p = job['params']
    positive = {'class_type': 'CLIPTextEncode', 'inputs': {'text': p['prompt'], 'clip': ['clip', 0]}}
    negative = {'class_type': 'CLIPTextEncode', 'inputs': {'text': p['negative_prompt'], 'clip': ['clip', 0]}}
    sampler = {'class_type': 'KSampler', 'inputs': {
        'model': ['model', 0], 'positive': ['positive', 0], 'negative': ['negative', 0],
        'latent_image': ['latent', 0], 'seed': p['seed'], 'steps': p['steps'],
        'cfg': p['guidance'], 'sampler_name': 'euler', 'scheduler': 'normal', 'denoise': 1.0,
    }}
    if job['model'] == 'sdxl':
        positive['inputs']['clip'] = ['model', 1]
        negative['inputs']['clip'] = ['model', 1]
        return {
            'model': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'sd_xl_base_1.0.safetensors'}},
            'positive': positive, 'negative': negative,
            'latent': {'class_type': 'EmptyLatentImage', 'inputs': {'width': p['width'], 'height': p['height'], 'batch_size': 1}},
            'sample': sampler,
            'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['model', 2]}},
            'save': {'class_type': 'SaveImage', 'inputs': {'images': ['decode', 0], 'filename_prefix': f"aigc/{job['id']}"}},
        }
    latent = {'class_type': 'Wan22ImageToVideoLatent', 'inputs': {
        'width': p['width'], 'height': p['height'], 'length': p['frames'], 'batch_size': 1, 'vae': ['vae', 0],
    }}
    graph = {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'wan2.2_ti2v_5B_fp16.safetensors', 'weight_dtype': 'default'}},
        'model': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['unet', 0], 'shift': 8.0}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'type': 'wan', 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'wan2.2_vae.safetensors'}},
        'positive': positive, 'negative': negative, 'latent': latent, 'sample': sampler,
        'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}},
        'video': {'class_type': 'CreateVideo', 'inputs': {'images': ['decode', 0], 'fps': float(p['fps'])}},
        'save': {'class_type': 'SaveVideo', 'inputs': {'video': ['video', 0], 'filename_prefix': f"aigc/{job['id']}", 'format': 'mp4', 'codec': 'h264'}},
    }
    if p['mode'] == 'image-to-video':
        if not uploaded_name:
            raise ValueError('参考图片尚未上传至推理服务')
        graph['image'] = {'class_type': 'LoadImage', 'inputs': {'image': uploaded_name}}
        latent['inputs']['start_image'] = ['image', 0]
    return graph

