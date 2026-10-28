"""One-time low-priority transfer of probe data off the home filesystem."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

SOURCE=Path('/home/zjr/Tracker/visual_evidence_probe/outputs')
STORE=Path('/data/zjr/visual_evidence_probe')


def inventory(root):
    files={};links={}
    for base,dirs,names in os.walk(root):
        for name in list(dirs):
            p=Path(base)/name
            if p.is_symlink():links[str(p.relative_to(root))]=os.readlink(p);dirs.remove(name)
        for name in names:
            p=Path(base)/name;key=str(p.relative_to(root))
            if p.is_symlink():links[key]=os.readlink(p)
            else:files[key]=p.stat().st_size
    return files,links


def transfer(source,dest,label):
    dest.mkdir(parents=True,exist_ok=True)
    log=STORE/'migration'/f'{label}.log'
    print(json.dumps(dict(stage='copy',label=label,source=str(source),dest=str(dest))),flush=True)
    with log.open('w') as output:
        subprocess.run(['/usr/bin/ionice','-c','3','/usr/bin/rsync','-a','--update','--bwlimit=65536',
            '--info=progress2',str(source)+'/',str(dest)+'/'],stdout=output,stderr=subprocess.STDOUT,check=True)


def main():
    os.nice(10)
    STORE.mkdir(parents=True,exist_ok=True);(STORE/'migration').mkdir(exist_ok=True)
    target=STORE/'outputs';stage=STORE/'outputs_migrating'
    stage.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(stage,target_is_directory=True)
    if target.is_symlink() and target.resolve()!=stage.resolve():raise ValueError('unexpected staging link')
    # Put all small inputs needed by the completed experiment on /data first.
    with (STORE/'migration/metadata.log').open('w') as output:
        subprocess.run(['/usr/bin/ionice','-c','3','/usr/bin/rsync','-a','--update','--bwlimit=65536',
            '--max-size=8388608','--include=*/','--include=*.json','--include=*.md','--include=*.jpg',
            '--include=*.png','--include=*.npz','--exclude=*',str(SOURCE)+'/',str(stage)+'/'],
            stdout=output,stderr=subprocess.STDOUT,check=True)
    aliases={}
    run=stage/'visible_error_localization_20261026'
    for index in range(4):
        root=run/'objects'/f'{index:03d}'
        obs=json.loads((root/'observations.json').read_text())
        bindings=obs['images']+obs['masks']+[json.loads((root/'evaluation_target.json').read_text())['mesh']]
        for b in bindings:
            old=Path(b['path'])
            if not str(old).startswith('/home/'):continue
            dest=STORE/'inputs'/f'{index:03d}'/old.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old,dest);aliases[str(old)]=str(dest)
    cache=STORE/'cache'
    for d in ('tmp','cache/matplotlib','cache/torch_extensions','cache/huggingface/hub'):(STORE/d).mkdir(parents=True,exist_ok=True)
    initial=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_link=str(SOURCE),
        physical_outputs=str(target),input_aliases=aliases,cache_root=str(cache),tmp=str(STORE/'tmp'),
        metadata_ready=True,migration_status='historical_outputs_copying',cache_ready=False)
    (STORE/'io_relocation.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(stage='metadata_ready',receipt=str(STORE/'io_relocation.json'))),flush=True)
    if SOURCE.is_symlink():
        if SOURCE.resolve()!=target.resolve():raise ValueError('unexpected existing output link')
    else:
        if target.exists() and not target.is_symlink():raise ValueError('destination collision')
        before=inventory(SOURCE)
        transfer(SOURCE,stage,'outputs')
        after=inventory(stage)
        for name,size in before[0].items():
            if after[0].get(name)!=size and Path(name).name not in ('REPORT.md','status.json','validation_receipt.json'):
                raise ValueError(f'copied file size differs: {name}; source retained')
        if any(after[1].get(k)!=v for k,v in before[1].items()):raise ValueError('copied links differ')
        # rsync validates reconstruction checksums during transfer. Avoid a second full home read.
        if target.is_symlink():target.unlink()
        stage.rename(target)
        backup=SOURCE.with_name('outputs_migrated_backup_20261026')
        if backup.exists():raise ValueError('backup collision')
        SOURCE.rename(backup)
        SOURCE.symlink_to(target,target_is_directory=True)
        observed=inventory(SOURCE)
        if not set(before[0]).issubset(observed[0]):raise ValueError('compatibility link failed; backup retained')
        shutil.rmtree(backup)
        print(json.dumps(dict(stage='outputs_moved',files=len(before[0]),bytes=sum(before[0].values()))),flush=True)
    for src,dst,label in [
        (Path('/home/zjr/.cache/torch/hub/facebookresearch_dinov2_main'),cache/'torch/hub/facebookresearch_dinov2_main','dino_code'),
        (Path('/home/zjr/.cache/huggingface/hub/models--Stable-X--trellis-vggt-v0-2'),cache/'huggingface/hub/models--Stable-X--trellis-vggt-v0-2','trellis_cache')]:
        if src.exists():transfer(src,dst,label)
    checkpoint=Path('/home/zjr/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth')
    if checkpoint.exists():
        dest=cache/'torch/hub/checkpoints';dest.mkdir(parents=True,exist_ok=True)
        log=STORE/'migration/dino_weights.log'
        with log.open('w') as output:
            subprocess.run(['/usr/bin/ionice','-c','3','/usr/bin/rsync','-a','--bwlimit=65536','--info=progress2',
                str(checkpoint),str(dest)+'/'],stdout=output,stderr=subprocess.STDOUT,check=True)
    for d in ('tmp','cache/matplotlib','cache/torch_extensions','cache/huggingface/hub'):(STORE/d).mkdir(parents=True,exist_ok=True)
    extensions=target/'torch_extensions'
    if extensions.exists():
        # Same /data filesystem: one rename, preserve the historical relative cache path.
        ext_dest=cache/'torch_extensions'
        if not any(ext_dest.iterdir()):
            ext_dest.rmdir();extensions.rename(ext_dest);extensions.symlink_to(ext_dest,target_is_directory=True)
    receipt=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_link=str(SOURCE),
        physical_outputs=str(target),input_aliases=aliases,
        cache_root=str(cache),tmp=str(STORE/'tmp'),verification='rsync transfer checksum plus complete size/link inventory',
        home_outputs_removed=True,shared_home_model_cache_untouched=True,
        metadata_ready=True,migration_status='complete',cache_ready=True,
        policy='all new probe data, results, caches and temporary files on /data/zjr; home source code only')
    (STORE/'io_relocation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(stage='complete',receipt=str(STORE/'io_relocation.json'))),flush=True)


if __name__=='__main__':main()
