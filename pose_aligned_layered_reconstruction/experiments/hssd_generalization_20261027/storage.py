"""One-time low-priority migration of target-encoder assets; keep home originals."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil

from . import common as c


def copy_verified(source, destination, expected):
    source,destination=Path(source),c.data_path(destination)
    if destination.exists():
        if c.sha(destination)!=expected:raise ValueError('Staged model mismatch: '+str(destination))
        return c.binding(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix(destination.suffix+'.partial')
    h=hashlib.sha256()
    with source.open('rb') as src,temp.open('wb') as dst:
        for chunk in iter(lambda:src.read(4<<20),b''):h.update(chunk);dst.write(chunk)
    if h.hexdigest()!=expected:raise ValueError('Original model differs from audited encoder')
    temp.replace(destination)
    return {'path':str(destination),'sha256':expected}


def stage():
    try:os.nice(10)
    except OSError:pass
    c.environment()
    prior=c.read(c.ROOT/'pose_aligned_layered_reconstruction/outputs/hssd_pilot_20261020/plan.json')['plan']['objects'][0]
    audit=c.read(c.ROOT/'pose_aligned_layered_reconstruction/outputs/hssd_pilot_20261020/objects'/prior['uid']/'latent_audit.json')
    old_dino=c.read(c.ROOT/'pose_aligned_layered_reconstruction/outputs/hssd_input_adaptation_20261025/encoder_identity.json')
    from huggingface_hub import try_to_load_from_cache
    local=c.DATASET/'models/TRELLIS-target-codec/ckpts';bindings=[]
    for suffix,key in [('.json','config_sha256'),('.safetensors','weights_sha256')]:
        name='ss_enc_conv3d_16l8_fp16'+suffix
        source=try_to_load_from_cache('microsoft/TRELLIS-image-large','ckpts/'+name,cache_dir='/home/zjr/.cache/huggingface/hub')
        if not isinstance(source,str):raise FileNotFoundError('Missing previously used SS encoder: '+name)
        bindings.append(copy_verified(source,local/name,audit['ss']['encoder'][key]))
    hub=c.DATA/'cache/torch/hub';dino_dest=hub/'checkpoints/dinov2_vitl14_reg4_pretrain.pth'
    bindings.append(copy_verified(old_dino['weights']['path'],dino_dest,old_dino['weights']['sha256']))
    repository=Path(old_dino['weights']['path']).parent.parent/'facebookresearch_dinov2_main'
    destination=hub/'facebookresearch_dinov2_main'
    for relative in ['dinov2','hubconf.py','LICENSE']:
        src=repository/relative;dst=destination/relative
        if src.is_dir():
            shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
        elif src.is_file():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    # Verify every frozen DINO source by its relative path in the staged repo.
    checked=0
    for row in old_dino['sources']:
        path=Path(row['path'])
        if path.is_relative_to(repository):
            copied=destination/path.relative_to(repository)
            if c.sha(copied)!=row['sha256']:raise ValueError('DINO source changed during staging')
            checked+=1
    result={'success':True,'models':bindings,'dino_sources_checked':checked,
            'local_ss_encoder':str(local.parent),'local_dino_repository':str(destination),
            'home_originals_retained':True,'source_pilot_audit':str(c.ROOT/'pose_aligned_layered_reconstruction/outputs/hssd_pilot_20261020/objects'/prior['uid']/'latent_audit.json')}
    c.locked(c.OUT/'encoder_storage.json',result);c.event('encoder_storage_ready',models=len(bindings),dino_sources=checked)


if __name__=='__main__':argparse.ArgumentParser(description=__doc__).parse_args();stage()
