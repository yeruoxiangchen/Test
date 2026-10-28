"""Copy a bounded reconstruction source snapshot, without touching Tracker Git.

The destination must be below /data/zjr/pose_constrained_trellis2/publication.
Model checkpoints, experiment products, raw datasets, and repository history
are intentionally not part of the source snapshot.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

SOURCE = Path('/home/zjr/Tracker')
DEST_ROOT = Path('/data/zjr/pose_constrained_trellis2/publication')
ROOTS = ['pose_constrained_trellis2','pose_aligned_layered_reconstruction',
         'pose_aligned_reconstruction','ar_ss_flow','reconvggt_ar_adapter_a','pixal3d_multiview',
         'datasets/scene_occlusion_pilot_20261020/render_tools']
EXTRA = ['experiments/rigid_aligned_mesh_eval_20260911.py']
SKIP = {'outputs','output','outputs2','results','result','datasets','data','cache','.cache',
        'checkpoints','ckpts','logs','runs','weights','__pycache__','.git','.pytest_cache',
        '.venv','venv','node_modules','external','tmp','temp'}
EXTENSIONS = {'.py','.sh','.md','.txt','.json','.yaml','.yml','.toml','.cjs','.js',
              '.html','.css','.sha256','.wasm'}
SECRETS = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}')
DEPENDENCIES = [
    ('TRELLIS2','https://github.com/microsoft/TRELLIS.2.git'),
    ('ReconViaGen','https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git'),
    ('pose_aligned_layered_reconstruction/external/GENAmodal3D','https://github.com/Colezwhy/GENAmodal3D.git'),
]


def git(repo,*args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(repo),*args])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    args=parser.parse_args();dest=Path(args.destination).resolve()
    if not dest.is_relative_to(DEST_ROOT):raise ValueError('Use project /data publication directory')
    dest.mkdir(parents=True,exist_ok=False)
    selected=[];skipped=[]
    for root in ROOTS:
        for directory,dirs,files in os.walk(SOURCE/root,followlinks=False):
            dirs[:]=[name for name in dirs if name not in SKIP and not Path(directory,name).is_symlink()]
            for name in files:
                p=Path(directory,name)
                allowed=(p.suffix.lower() in EXTENSIONS or name in {'.gitignore','LICENSE','NOTICE'} or name.endswith('_LICENSE'))
                if not allowed or p.is_symlink():
                    skipped.append(str(p.relative_to(SOURCE)));continue
                selected.append(p)
    selected.extend(SOURCE/name for name in EXTRA)
    manifest=[]
    for source in sorted(set(selected)):
        before=source.stat();data=source.read_bytes();after=source.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise RuntimeError('Source changed while copying: '+str(source))
        if len(data)>8*1024*1024:raise ValueError('Unexpected large source file: '+str(source))
        if SECRETS.search(data):raise ValueError('Credential-shaped content requires review: '+str(source))
        if source.suffix=='.py':ast.parse(data,filename=str(source))
        target=dest/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data);os.chmod(target,0o755 if before.st_mode&0o111 else 0o644)
        manifest.append({'path':str(source.relative_to(SOURCE)),'bytes':len(data),
                         'sha256':hashlib.sha256(data).hexdigest()})
    dependencies=[]
    for relative,url in DEPENDENCIES:
        repo=SOURCE/relative
        if Path(git(repo,'rev-parse','--show-toplevel').decode().strip()).resolve()!=repo.resolve():raise ValueError('Not a standalone dependency repo')
        commit=git(repo,'rev-parse','HEAD').decode().strip()
        dependencies.append({'path':relative,'url':url,'commit':commit})
    # Preserve only reconstruction-related changes in this external checkout.
    patch=git(SOURCE/'ReconViaGen','diff','HEAD','--','trellis/pipelines/trellis_image_to_3d.py','trellis/representations/mesh/flexicube.py')
    patches=dest/'pose_constrained_trellis2/publication/dependency_patches';patches.mkdir(parents=True,exist_ok=True)
    (patches/'reconviagen_geometry.patch').write_bytes(patch)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'source_repository_head':git(SOURCE,'rev-parse','HEAD').decode().strip(),
            'scope':ROOTS+EXTRA,'source_files':manifest,'source_file_count':len(manifest),'source_bytes':sum(r['bytes'] for r in manifest),
            'dependencies':dependencies,'dependency_patch':'pose_constrained_trellis2/publication/dependency_patches/reconviagen_geometry.patch',
            'exclusions':'runtime outputs, datasets, weights, caches, other conversations, original Git history',
            'skipped_non_source_files':skipped}
    (dest/'RECONSTRUCTION_SNAPSHOT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (dest/'RECONSTRUCTION_README.md').write_text('''# 多视图遮挡物体重建：代码与实验记录快照

这是Tracker中本条重建研究主线的独立源码快照。主线为TRELLIS.2多视图独立物体基线及场景关系实验准备；保留LaRI/HSSD实验历史及必要旧源码依赖。

- [当前基线与监视说明](pose_constrained_trellis2/experiments/hssd_mv_baseline_20261028/README.md)
- [原生标签、坐标与真实mesh审阅](pose_constrained_trellis2/experiments/hssd_scene_20261028/REVIEW_20261028.md)
- [后续关系重建设计](pose_constrained_trellis2/SCENE_RELATIONS_PLAN_20261028.md)
- [按时间记录的实验审计](pose_aligned_layered_reconstruction/审计.md)
- [源码文件校验及外部版本](RECONSTRUCTION_SNAPSHOT.json)

## 外部代码

TRELLIS2、ReconViaGen及GENAmodal3D以固定commit的Git子模块记录，保留各自许可。

```bash
git submodule update --init --recursive
git -C ReconViaGen apply ../pose_constrained_trellis2/publication/dependency_patches/reconviagen_geometry.patch
```

ReconViaGen补丁仅包含本重建实验使用的SS支持导出及FlexiCubes索引检查，不含其他应用界面改动。TRELLIS.2原生主线不依赖此旧后端补丁。

## 数据与复现边界

此仓库不包含HSSD/ScanNet等原始数据、模型权重、latent缓存、检查点、渲染图或实验输出。它们保留在服务器/data；文档中的/data和/home路径记录原运行位置，在其他机器上不可直接访问。

冻结实验路径和协议原样保存；换机器需要按数据准备及依赖说明准备环境和路径映射，不应修改旧报告冒充原运行。依赖见各项目requirements及Native12运行环境记录。当前只验证了原生codec和短运行，MVBase12的训练状态需以原服务器监视为准，不能把源码发布当成关系模型已经完成训练。

源文件为创建快照时的内容，校验值见JSON。发布在独立分支，未修改原Tracker工作区的分支或暂存区，也没有将其他对话未提交的业务项目纳入快照。
''',encoding='utf-8')
    # Detect concurrent edits to the selected small sources; never stage a torn snapshot.
    changed=[r['path'] for r in manifest if hashlib.sha256((SOURCE/r['path']).read_bytes()).hexdigest()!=r['sha256']]
    if changed:raise RuntimeError('Source changed during snapshot: '+', '.join(changed))
    print(json.dumps({'destination':str(dest),'files':len(manifest),'bytes':record['source_bytes'],'dependencies':dependencies},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
