import hashlib
import shutil

from .config import OUT, STORE, PRIOR, OMNI, PARAMS, folder, read, write, now, bind_engine, setup, register


def select_rows(objects, excluded, count=16):
    def order(value):
        return hashlib.sha256(f'{PARAMS["selection_seed"]}|{value}'.encode()).hexdigest()
    eligible = [r for r in objects if r['uid'] not in excluded]
    categories = sorted({r['category'] for r in eligible}, key=lambda c: (order('category|' + c), c))[:count]
    if len(categories) != count:
        raise ValueError('not enough eligible categories')
    return [min((r for r in eligible if r['category'] == c), key=lambda r: (order('uid|' + r['uid']), r['uid']))
            for c in categories]


def main():
    bind_engine(); setup()
    from pose_aligned_layered_reconstruction._bootstrap import bootstrap_external_paths
    bootstrap_external_paths()
    from visual_evidence_probe.experiments.dense_evidence.prepare import prepare_row
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(PRIOR / 'registry.json')
    excluded = {r['uid'] for r in old['rows'] if r['kind'] == 'synthetic_clean'}
    manifest = read(OMNI / 'manifest.json')
    chosen = select_rows(manifest['objects'], excluded)
    rows = [dict(index=i, uid=r['uid'], category=r['category'], split='validation',
                 kind='synthetic_clean', source=r) for i, r in zip(PARAMS['indices'], chosen)]
    value = dict(time=now(), rows=rows, excluded_uids=sorted(excluded), source_manifest=str(OMNI / 'manifest.json'),
                 selection='fixed hashed category/UID order, no generated quality or GT scores',
                 source_registry=str(PRIOR / 'registry.json'), previously_inspected_by_repair_development=False)
    if (OUT / 'registry.json').exists():
        if read(OUT / 'registry.json')['rows'] != rows:
            raise ValueError('frozen registry changed')
    else:
        write(OUT / 'registry.json', value)
    for r in rows:
        prepare_row(r)
        for seed in PARAMS['seeds']:
            folder(r['index'], seed).mkdir(parents=True, exist_ok=True)
        print(dict(stage='input_prepared', index=r['index'], uid=r['uid']), flush=True)
    for name in ('da3_weights.json', 'da3_environment.json'):
        if (PRIOR / name).exists() and not (OUT / name).exists():
            shutil.copyfile(PRIOR / name, OUT / name)
    register('prepare')
    write(OUT / 'preparation_complete.json', dict(time=now(), objects=len(rows), cases=len(rows)*2,
          disjoint_from_previous_repair_cohorts=not bool(excluded & {r['uid'] for r in rows}),
          dataset='OmniObject3D scans with four pre-rendered posed RGB inputs'))
    audit = STORE / '审计.md'; marker = '# 10 第十轮：自然生成错误的纠正任务启动'
    before = audit.read_bytes()
    if marker.encode() not in before:
        note = f'\n\n{marker}\n\n时间：{now()}。\n\n'
        note += '前九轮作为开发记录保留。第十轮冻结既有完整输入驱动纠错方案，重点检验相对base的收益和原错误区域修复能力。\n'
        note += '16个与既有局部修复开发/验证物体不重合的Omni物体，每个2个seed；4张已知相机渲染输入。全部案例运行并保留回退。\n'
        note += '原始30K、输入对齐30K、完整纠错结果分别报告。GT仅在输入决策封存后评估；高/低错误分层不进入优化。\n'
        note += f'技术smoke先跑首个物体的两个seed，通过后后台继续其余物体；不以涨点作为smoke通过条件。\n\n输出：`{OUT}`。\n'
        with audit.open('ab') as stream:
            stream.write(note.encode())
        if not audit.read_bytes().startswith(before):
            raise RuntimeError('prior audit content changed')


if __name__ == '__main__':
    main()
