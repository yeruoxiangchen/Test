from __future__ import annotations

import hashlib
import os
from pathlib import Path

from video_count_probe.core import atomic_json, digest, file_sha, read_json, writer_lock
from video_count_probe.downloads import retry_hub_call
from .protocol import MODEL_ID, under_data

MANIFEST = 'wan22_download_manifest.json'
RECEIPT = 'wan22_model_verified.json'


def selected(name):
    if name == 'model_index.json':
        return True
    return (name.startswith(('scheduler/', 'tokenizer/', 'text_encoder/', 'transformer/', 'vae/'))
            and name.endswith(('.json', '.safetensors', '.model', '.txt'))
            and not any(v in name for v in ('.fp16.', '.bf16.', '.fp32.')))


def validate_manifest(manifest):
    if manifest['model_id'] != MODEL_ID or len(manifest['revision']) != 40 or any(c not in '0123456789abcdef' for c in manifest['revision']):
        raise ValueError('Wrong model or revision')
    paths = set()
    for item in manifest['files']:
        p = Path(item['path'])
        if p.is_absolute() or '..' in p.parts or not selected(str(p)) or str(p) in paths:
            raise ValueError('Unsafe or duplicate model path')
        paths.add(str(p))
        if item['size'] <= 0 or (str(p).endswith('.safetensors') and not item['sha256']):
            raise ValueError('Model weights require remote size and SHA-256')
    required = {'model_index.json', 'vae/config.json', 'transformer/config.json',
                'scheduler/scheduler_config.json', 'text_encoder/config.json', 'tokenizer/tokenizer_config.json'}
    if not required <= paths or any(not any(p.startswith(part + '/') and p.endswith('.safetensors') for p in paths)
                                    for part in ('vae', 'transformer', 'text_encoder')):
        raise ValueError('Incomplete Diffusers model manifest')


def get_manifest(model_dir, attempts):
    from huggingface_hub import HfApi
    path = model_dir / MANIFEST
    if path.exists():
        result = read_json(path)
        validate_manifest(result)
        return result
    info = retry_hub_call(lambda: HfApi(token=False).model_info(MODEL_ID, files_metadata=True, timeout=30),
                          'repository metadata', model_dir / 'download_status.json', attempts=attempts)
    files = []
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        if not selected(s.rfilename):
            continue
        sha = (s.lfs.get('sha256') if isinstance(s.lfs, dict) else getattr(s.lfs, 'sha256', None)) if s.lfs else None
        files.append({'path': s.rfilename, 'size': s.size, 'sha256': sha, 'git_blob_sha1': s.blob_id})
    result = {'model_id': MODEL_ID, 'revision': info.sha, 'files': files}
    validate_manifest(result)
    atomic_json(path, result)  # Pin BEFORE any file transfer, including partial downloads.
    return result


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected['size']:
        raise ValueError(f'Missing or wrong-sized model file: {path}')
    sha = file_sha(path)
    if expected.get('sha256'):
        if sha != expected['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')
    elif expected.get('git_blob_sha1'):
        data = path.read_bytes()
        git_sha = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if git_sha != expected['git_blob_sha1']:
            raise ValueError(f'Git blob hash mismatch: {path}')
    return {'path': expected['path'], 'size': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha}


def download(model_dir, *, metadata_only=False, attempts=5):
    from huggingface_hub import hf_hub_download
    model_dir = under_data(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(model_dir / '.download.lock'):
        manifest = get_manifest(model_dir, attempts)
        files = [f for f in manifest['files'] if not metadata_only or not f['path'].endswith('.safetensors')]
        print(f'Pinned {MODEL_ID}@{manifest["revision"]}; full model {sum(f["size"] for f in manifest["files"])/1e9:.2f} GB; selected {sum(f["size"] for f in files)/1e6:.1f} MB', flush=True)
        verified_path = model_dir / ('wan22_metadata_verified.json' if metadata_only else RECEIPT)
        previous = read_json(verified_path) if verified_path.exists() else {}
        previous = {v['path']: v for v in previous.get('files', [])} if previous.get('manifest_hash') == digest(manifest) else {}
        verified = []
        for i, expected in enumerate(files, 1):
            path = model_dir / expected['path']
            old = previous.get(expected['path'])
            if old and path.is_file() and path.stat().st_size == old['size'] and path.stat().st_mtime_ns == old['mtime_ns']:
                verified.append(old)
                continue
            print(f'[{i}/{len(files)}] {expected["path"]}', flush=True)
            retry_hub_call(lambda: hf_hub_download(MODEL_ID, expected['path'], revision=manifest['revision'],
                                                   local_dir=str(model_dir), token=False, etag_timeout=30),
                           expected['path'], model_dir / 'download_status.json', attempts=attempts)
            verified.append(verify_file(path, expected))
        receipt = {'model_id': MODEL_ID, 'revision': manifest['revision'],
                   'manifest_hash': digest(manifest), 'files': verified}
        validate_configs(model_dir)
        atomic_json(verified_path, receipt)
        atomic_json(model_dir / 'download_status.json', {'state': 'complete', 'metadata_only': metadata_only,
                                                       'revision': manifest['revision'], 'verified_files': len(files)})
        print(f'Verified: {verified_path}', flush=True)


def model_receipt(model_dir, metadata_only=False):
    model_dir = Path(model_dir)
    manifest = read_json(model_dir / MANIFEST)
    validate_manifest(manifest)
    name = 'wan22_metadata_verified.json' if metadata_only else RECEIPT
    path = model_dir / name
    if not path.exists():
        raise RuntimeError(f'Model not verified: {path}; run metadata/download explicitly first')
    receipt = read_json(path)
    wanted = {f['path'] for f in manifest['files'] if not metadata_only or not f['path'].endswith('.safetensors')}
    if receipt['model_id'] != MODEL_ID or receipt['revision'] != manifest['revision'] or receipt['manifest_hash'] != digest(manifest):
        raise ValueError('Receipt/manifest mismatch')
    if len(receipt['files']) != len(wanted) or {r['path'] for r in receipt['files']} != wanted:
        raise ValueError('Incomplete verification receipt')
    for item in receipt['files']:
        p = model_dir / item['path']
        if not p.is_file() or p.stat().st_size != item['size'] or p.stat().st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'Model file changed: {p}; rerun download to verify')
    return receipt


def validate_configs(model_dir):
    model_dir = Path(model_dir)
    idx = read_json(model_dir / 'model_index.json')
    tr = read_json(model_dir / 'transformer/config.json')
    vae = read_json(model_dir / 'vae/config.json')
    sched = read_json(model_dir / 'scheduler/scheduler_config.json')
    if idx['_class_name'] != 'WanPipeline' or idx.get('expand_timesteps') is not True or idx.get('boundary_ratio') is not None:
        raise ValueError('Expected single-transformer Wan2.2 TI2V pipeline with expanded timesteps')
    if (tr['in_channels'], tr['out_channels'], tr['num_layers'], tr['num_attention_heads'], tr['attention_head_dim']) != (48, 48, 30, 24, 128):
        raise ValueError('Unexpected 5B transformer architecture')
    if tuple(tr['patch_size']) != (1, 2, 2) or tr.get('added_kv_proj_dim') is not None:
        raise ValueError('Unexpected transformer patch/image layout')
    if (vae['z_dim'], vae['scale_factor_temporal'], vae['scale_factor_spatial'], vae['patch_size']) != (48, 4, 16, 2):
        raise ValueError('Expected Wan2.2 48-channel 4x16x16 VAE')
    if vae.get('is_residual') is not True or vae['in_channels'] != 12 or vae['out_channels'] != 12:
        raise ValueError('Expected residual patchified Wan2.2 VAE')
    if sched['_class_name'] != 'UniPCMultistepScheduler' or sched['prediction_type'] != 'flow_prediction' or not sched['use_flow_sigmas'] or sched['flow_shift'] != 5.0:
        raise ValueError('Unexpected official scheduler settings')
    return {'pipeline': idx, 'transformer': tr, 'vae': vae, 'scheduler': sched}


def text_identity(model_dir):
    from video_count_probe.runtime import versions
    receipt = model_receipt(model_dir)
    return {'model_id': MODEL_ID, 'revision': receipt['revision'], 'packages': versions(),
            'files': {r['path']: r['sha256'] for r in receipt['files'] if r['path'].startswith(('tokenizer/', 'text_encoder/'))},
            'dtype': 'bfloat16', 'device': 'cpu', 'max_length': 512, 'padding': 'WanPipeline._get_t5_prompt_embeds'}


def cache_path(root, prompt, identity):
    return Path(root) / 'text_cache' / (digest({'prompt': prompt, 'identity': identity}) + '.pt')


def read_embedding(path, prompt, identity):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    x = value['embedding']
    if value['identity'] != identity or value['prompt'] != prompt or tuple(x.shape) != (1, 512, 4096) or x.dtype != torch.bfloat16 or not torch.isfinite(x).all():
        raise ValueError(f'Invalid prompt cache: {path}')
    if value['tensor_sha256'] != hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest():
        raise ValueError(f'Changed cached embedding: {path}')
    return x


def encode(root, model_dir, prompts):
    from types import SimpleNamespace
    import torch
    from diffusers import WanPipeline
    from transformers import T5TokenizerFast, UMT5EncoderModel
    identity = text_identity(model_dir)
    with writer_lock(Path(root) / '.text_cache.lock'):
        pending = []
        for prompt in dict.fromkeys(prompts):
            path = cache_path(root, prompt, identity)
            if path.exists():
                read_embedding(path, prompt, identity)
            else:
                pending.append(prompt)
        if not pending:
            print('All CPU text embeddings already verified; text model will not load.', flush=True)
            return
        print(f'Encoding {len(pending)} prompts on CPU. No GPU model will load.', flush=True)
        tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
        encoder = UMT5EncoderModel.from_pretrained(model_dir, subfolder='text_encoder', torch_dtype=torch.bfloat16, local_files_only=True).eval()
        holder = SimpleNamespace(tokenizer=tokenizer, text_encoder=encoder)
        for i, prompt in enumerate(pending, 1):
            from diffusers.pipelines.wan.pipeline_wan import prompt_clean
            if len(tokenizer(prompt_clean(prompt), truncation=False).input_ids) > 512:
                raise ValueError('Prompt would be truncated')
            with torch.no_grad():
                embedding = WanPipeline._get_t5_prompt_embeds(holder, prompt, max_sequence_length=512,
                                                             device=torch.device('cpu'), dtype=torch.bfloat16).contiguous()
            path = cache_path(root, prompt, identity)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f'.{os.getpid()}.partial')
            torch.save({'identity': identity, 'prompt': prompt, 'embedding': embedding,
                        'tensor_sha256': hashlib.sha256(embedding.view(torch.uint8).numpy().tobytes()).hexdigest()}, temporary)
            os.replace(temporary, path)
            read_embedding(path, prompt, identity)
            print(f'CPU prompt cache {i}/{len(pending)} ready', flush=True)
