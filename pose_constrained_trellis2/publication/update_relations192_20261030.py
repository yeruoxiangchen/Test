"""Publish only the current reconstruction runtime sources and experiment notes."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/home/zjr/Tracker')
DEST = Path('/data/zjr/pose_constrained_trellis2/publication/Test_relations192_20261030')
OUT = Path('/data/zjr/pose_constrained_trellis2/outputs/hssd_relations192_20261029')


def main():
    spec = importlib.util.spec_from_file_location('export', SOURCE / 'pose_constrained_trellis2/publication/export_snapshot.py')
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
    assert not subprocess.check_output(['git', '-C', str(DEST), 'status', '--porcelain']).strip(), 'Publication checkout must be clean'
    runtime = json.loads((OUT / 'runtime.json').read_text())
    paths = set(runtime['runtime']['files'])
    root = SOURCE / 'pose_constrained_trellis2/experiments/hssd_relations192_20261029'
    paths.update(str(p.relative_to(SOURCE)) for p in root.rglob('*') if p.is_file() and p.suffix in {'.py', '.sh', '.md'} and '__pycache__' not in p.parts)
    paths.update(['pose_constrained_trellis2/SCENE_RELATIONS_PLAN_20261028.md',
                  'pose_aligned_layered_reconstruction/审计.md',
                  'pose_constrained_trellis2/publication/update_relations192_20261030.py'])
    records = []
    for relative in sorted(paths):
        src = SOURCE / relative
        assert not src.is_symlink() and src.suffix in {'.py', '.sh', '.md'}
        data = src.read_bytes(); assert len(data) < 8 * 1024**2
        assert not exporter.SECRETS.search(data), f'Review credential-shaped content: {relative}'
        digest = hashlib.sha256(data).hexdigest()
        if relative in runtime['runtime']['files']: assert digest == runtime['runtime']['files'][relative], relative
        if src.suffix == '.py': ast.parse(data, filename=relative)
        if src.suffix == '.sh': subprocess.run(['bash', '-n', str(src)], check=True)
        dst = DEST / relative; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data); shutil.copymode(src, dst)
        records.append({'path': relative, 'bytes': len(data), 'sha256': digest})
    now = datetime.now(timezone.utc).isoformat()
    record = {'created_utc': now, 'runtime_identity': runtime['identity'], 'source_files': records,
              'scope': 'Current Relations192 runtime source closure, operations, protocol and small research notes',
              'exclusions': 'Datasets, model weights, checkpoints, latent caches, rendered assets and other conversations',
              'experiment_status': 'Data construction complete; formal training in progress; final mesh evaluation not yet complete'}
    (DEST / 'RECONSTRUCTION_UPDATE_20261030.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    manifest_path = DEST / 'RECONSTRUCTION_SNAPSHOT.json'
    manifest = json.loads(manifest_path.read_text())
    merged = {r['path']: r for r in manifest['source_files']}; merged.update({r['path']: r for r in records})
    manifest['source_files'] = [merged[k] for k in sorted(merged)]
    manifest['source_file_count'] = len(merged); manifest['source_bytes'] = sum(r['bytes'] for r in merged.values())
    manifest['updated_utc'] = now
    manifest['incremental_update'] = 'RECONSTRUCTION_UPDATE_20261030.json; other entries retain the original snapshot provenance'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    readme = DEST / 'RECONSTRUCTION_README.md'
    text = readme.read_text()
    text = text.replace('主线为TRELLIS.2多视图独立物体基线及场景关系实验准备', '主线为TRELLIS.2多视图独立物体基线及Relations192跨物体关系实验')
    text = text.replace('- [当前基线与监视说明]', '- [Relations192运行与监视](pose_constrained_trellis2/experiments/hssd_relations192_20261029/README.md)\n- [原生支持容量修复验收](pose_constrained_trellis2/experiments/hssd_relations192_20261029/TOKEN_SUPPORT_REPAIR_20261029.md)\n- [2026-10-30阶段性分析](pose_constrained_trellis2/experiments/hssd_relations192_20261029/INTERIM_REVIEW_20261030.md)\n- [本次增量来源与校验](RECONSTRUCTION_UPDATE_20261030.json)\n- [既有基线与监视说明]')
    text = text.replace('当前只验证了原生codec和短运行，MVBase12的训练状态需以原服务器监视为准，不能把源码发布当成关系模型已经完成训练。', 'Relations192的289物体数据构建和检查已完成，正式两臂训练正在进行；最终mesh评估尚未完成。各文档保留其记录时间，最新进度以原服务器监视为准，不能把源码发布当成最终模型效果。')
    readme.write_text(text)
    for row in records:
        assert hashlib.sha256((SOURCE / row['path']).read_bytes()).hexdigest() == row['sha256'], 'Concurrent source edit'
    stage = sorted(paths) + ['RECONSTRUCTION_UPDATE_20261030.json', 'RECONSTRUCTION_SNAPSHOT.json', 'RECONSTRUCTION_README.md']
    subprocess.run(['git', '-C', str(DEST), 'add', '--', *stage], check=True)
    print(json.dumps({'copied_source_files': len(records), 'bytes': sum(r['bytes'] for r in records), 'runtime_identity': runtime['identity']}))


if __name__ == '__main__': main()
