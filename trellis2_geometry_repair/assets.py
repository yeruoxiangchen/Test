"""Stage only required assets once, keeping the shared home cache untouched."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil

from .config import STORE, MODELS, OUT, read, write, now

REVISION='af44b45f2e35a493886929c6d786e563ec68364d'
HF=Path('/home/zjr/.cache/huggingface/hub')


def copy_verified(source,target):
    source=Path(source);target=Path(target);real=source.resolve(strict=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    expected=real.name if len(real.name)==64 and all(c in '0123456789abcdef' for c in real.name) else None
    if target.exists() and target.stat().st_size!=source.stat().st_size:
        raise RuntimeError(f'incomplete target exists: {target}')
    if not target.exists():
        if real.is_relative_to(Path('/data/zjr')):
            try:os.link(real,target)
            except OSError:shutil.copyfile(real,target)
        else:
            partial=target.with_name(target.name+'.partial')
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>source.stat().st_size:raise RuntimeError('invalid partial size')
            with real.open('rb') as src,partial.open('ab') as dest:
                src.seek(offset)
                shutil.copyfileobj(src,dest,length=4*1024*1024)
            partial.replace(target)
    # Verify on /data only; never hash a multi-GB source on home.
    h=hashlib.sha256()
    with target.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    observed=h.hexdigest()
    if expected and observed!=expected:raise RuntimeError(f'weight checksum mismatch: {target}')
    return dict(source=str(source),source_physical=str(real),path=str(target),bytes=target.stat().st_size,
                sha256=observed,expected_hf_blob_sha256=expected)


def main():
    MODELS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    lock=(MODELS/'staging.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    manifest=MODELS/'manifest.json'
    if manifest.exists():
        data=read(manifest)
        if not all(Path(r['path']).is_file() and Path(r['path']).stat().st_size==r['bytes'] for r in data['files']):
            raise RuntimeError('staged asset size/availability changed')
        print(dict(stage='assets_ready',reused_verified_receipt=True),flush=True);return
    t2=HF/'models--microsoft--TRELLIS.2-4B/snapshots'/REVISION
    ss=HF/'models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96'
    dino=HF/'models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881'
    files=[]
    files.append(copy_verified(t2/'pipeline.json',MODELS/'pipeline.json'))
    for name,root in [('ss_flow_img_dit_1_3B_64_bf16',t2),
                      ('slat_flow_img2shape_dit_1_3B_512_bf16',t2),
                      ('ss_dec_conv3d_16l8_fp16',ss),
                      ('shape_dec_next_dc_f16c32_fp16',t2)]:
        for suffix in ('.json','.safetensors'):
            candidate=Path('/data/zjr/models/microsoft_TRELLIS.2-4B_shape_vae/ckpts')/(name+suffix)
            source=candidate if candidate.exists() else root/'ckpts'/(name+suffix)
            print(dict(stage='staging_asset',name=name+suffix,source=str(source)),flush=True)
            item=copy_verified(source,MODELS/'ckpts'/(name+suffix));files.append(item)
    for name in ('config.json','model.safetensors'):
        print(dict(stage='staging_dino',name=name),flush=True)
        files.append(copy_verified(dino/name,MODELS/'dinov3'/name))
    write(manifest,dict(time=now(),revision=REVISION,files=files,
          dino_source='camenduru mirror of dinov3-vitl16-pretrain-lvd1689m; identity recorded, not silently substituted',
          original_shared_cache='retained unchanged',all_runtime_weights_on_data=True))
    print(dict(stage='assets_ready',files=len(files)),flush=True)


if __name__=='__main__':main()
