"""Reevaluate the preselected mesh4 cases under one frozen native GEMM policy.

Two independent processes reverse both case order and arm order. No SS support
is deduplicated, including uniform/balanced's identical shelf support. The long
72-mesh run is explicit and requires a two-process complete-path smoke pass.
"""
import argparse
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .common import OUT, DEPTH, MODULE, PYTHON, freeze, binding, verify, read, write, locked, now
from .kernel_policy import FixedNative
from ..hssd_mesh_replay_20261025.replay import save_tensor_file, sparse_payload, environment
from ..hssd_mesh_replay_20261025.tracing import tensor_meta
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from ...frozen_slat import (INPUT_KEYS, canonical_coords, configure_determinism,
    load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent, module_state_sha256)
from ...v2.dataset import SceneDataset, select_views
from ...mesh_metrics import surface_metrics, structure_metrics

FOUT = OUT / 'fair'
ARMS = ['appearance_control', 'uniform', 'balanced']
INDICES, SEEDS = [0, 2, 5, 14], [42, 43, 44]


def protocol():
    freeze()
    check = read(OUT / 'kernel_check.json')
    if not check['passed']:
        raise ValueError('The kernel intervention must pass first')
    kp = read(verify(check['protocol']))
    for p in kp['sources']: verify(p)
    source = read(DEPTH / 'protocol.json')
    value = {'format': 'hssd_mesh4_fixed_native.v1',
        'sources': [binding(Path(__file__).with_name(f)) for f in ['fair.py', 'fair_visuals.py', 'kernel_policy.py']],
        'operator_protocol': binding(OUT / 'protocol.json'), 'kernel_check': binding(OUT / 'kernel_check.json'),
        'manifest': source['manifest'], 'slat': binding(DEPTH / 'mesh/uniform/slat_binding.json'),
        'indices': INDICES, 'seeds': SEEDS, 'arms': ARMS, 'views': 8,
        'ss': {f'{i}/{s}/{a}': binding(DEPTH / 'evaluations' / a / f'{i:03d}' / f'{s}_ss.json')
            for i in INDICES for s in SEEDS for a in ARMS},
        'order': ['ascending cases, appearance_control/uniform/balanced',
                  'descending cases, balanced/uniform/appearance_control'],
        'mesh_count': 72, 'smoke': [[2, 42, 'uniform'], [2, 42, 'balanced']],
        'noise': 'original CUDA master_noise(global_index,seed), saved once and reused by every arm/process',
        'alignment': 'native coordinates, transform_pose=False, no fitted rigid/similarity/ICP transform',
        'precision': 'same fp16/bf16 as original; only native spconv descriptor selection is fixed',
        'scope': 'four previously selected training objects, one scene, three sampling seeds; no heldout or scaling claim',
        'resume': 'completed meshes are verified and reused; each new worker segment records its identity'}
    locked(FOUT / 'protocol.json', value)
    return value


@torch.no_grad()
def prepare():
    torch.set_num_threads(2)
    p = protocol()
    done = FOUT / 'inputs.json'
    if done.exists():
        info = read(done)
        if info['protocol'] != binding(FOUT / 'protocol.json'):
            raise ValueError('Changed prepared fair-evaluation identity')
        for rows in [info['samples'], info['noises'], info['ss']]:
            for row in rows.values():
                for key in ['payload', 'file', 'source', 'gt', 'coords']:
                    if key in row: verify(row[key])
        return info
    configure_determinism()
    dataset = SceneDataset(verify(p['manifest']), require_labels=False)
    info = {'protocol': binding(FOUT / 'protocol.json'), 'samples': {}, 'noises': {}, 'ss': {}}
    for index in INDICES:
        sample = select_views(dataset[index], torch.arange(8))
        payload = {'sample': {k: sample[k] for k in INPUT_KEYS},
            'uid': sample['uid'], 'object_uid': sample['object_uid']}
        info['samples'][str(index)] = {
            'payload': save_tensor_file(FOUT / 'inputs' / f'{index:03d}.pt', payload),
            'source': binding(dataset.rows[index]['sample']), 'gt': sample['target_mesh_binding']}
        verify(sample['target_mesh_binding'])
        for seed in SEEDS:
            noise = master_noise(seed, index, 'cuda:0').cpu()
            info['noises'][f'{index}/{seed}'] = {'file': save_tensor_file(
                FOUT / 'inputs' / f'{index:03d}_{seed}_noise.pt', noise), 'tensor': tensor_meta(noise)}
            for arm in ARMS:
                key = f'{index}/{seed}/{arm}'
                record = read(verify(p['ss'][key]))
                if (record['index'], record['seed'], record['arm'], record['object_uid'], record['success']) != (
                        index, seed, arm, sample['object_uid'], True):
                    raise ValueError('Changed SS identity: ' + key)
                with np.load(verify(record['coords'])) as data:
                    coords = canonical_coords(data['coords'])
                if binding(verify(record['gt']))['sha256'] != sample['target_mesh_binding']['sha256']:
                    raise ValueError('SS and scene input have different target geometry')
                info['ss'][key] = {'source': p['ss'][key], 'coords': record['coords'],
                    'tensor': tensor_meta(torch.from_numpy(coords))}
    locked(done, info)
    return info


def tasks(mode, repeat):
    if mode == 'smoke':
        return [(2, 42, ['uniform', 'balanced'][repeat])]
    cases = [(i, s) for i in INDICES for s in SEEDS]
    arms = ARMS
    if repeat == 1:
        cases, arms = list(reversed(cases)), list(reversed(arms))
    return [(i, s, a) for i, s in cases for a in arms]


def destination(mode, repeat, index, seed, arm):
    return FOUT / mode / str(repeat) / f'{index:03d}_{seed}_{arm}' / 'result.json'


def check_row(path):
    row = read(path)
    if row['protocol'] != binding(FOUT / 'protocol.json'):
        raise ValueError('Changed completed mesh protocol')
    for key in ['mesh', 'latent', 'initial', 'gt', 'source_ss', 'master_noise', 'input_payload']:
        verify(row[key])
    segment = read(verify(row['segment']))
    if row['weights'] != segment['weights'] or row['fitted_transform'] is not False:
        raise ValueError('Invalid weights/frame in completed mesh')
    if not row['success']:
        raise ValueError('Failed stored mesh')
    return row


@torch.no_grad()
def worker(mode, repeat):
    import trimesh
    torch.set_num_threads(2)
    protocol()
    info = read(FOUT / 'inputs.json')
    if info['protocol'] != binding(FOUT / 'protocol.json'):
        raise ValueError('Prepared inputs have a different protocol')
    todo = tasks(mode, repeat)
    remaining = []
    for i, s, a in todo:
        path = destination(mode, repeat, i, s, a)
        if path.exists():
            row = check_row(path)
            if (row['mode'], row['repeat'], row['index'], row['seed'], row['arm']) != (mode, repeat, i, s, a):
                raise ValueError('Wrong completed case identity')
        else: remaining.append((i, s, a))
    if not remaining: return
    patch = install_memory_efficient_mesh()
    runtime = load_frozen_slat30k(torch.device('cuda:0'))
    if runtime['binding'] != read(verify(read(FOUT / 'protocol.json')['slat'])):
        raise ValueError('Different frozen SLat binding')
    weights = {k: module_state_sha256(runtime[k]) for k in ['model', 'decoder']}
    expected = {'model': 'dba7957867d4bca06c21f47db5f6abfefd898cf2670fbdcc2d05d362087e0cd7',
        'decoder': '3372a1a95e874f7cad9d512d2bc6eda156637145e03dce9293a1c78cdfc03c03'}
    if weights != expected or any(m.training for k in weights for m in runtime[k].modules()):
        raise ValueError('Unexpected model weights or train mode')
    versions = {k: {n: p._version for n, p in runtime[k].named_parameters()} for k in weights}
    segment_path = FOUT / mode / str(repeat) / f'segment_{time.time_ns()}.json'
    write(segment_path, {'time': now(), 'protocol': binding(FOUT / 'protocol.json'),
        'mode': mode, 'repeat': repeat, 'remaining': remaining, 'weights': weights,
        'environment': environment(), 'patch': patch})
    with FixedNative() as fixed:
        for ordinal, (index, seed, arm) in enumerate(remaining):
            begin = time.monotonic()
            dest = destination(mode, repeat, index, seed, arm)
            dest.parent.mkdir(parents=True, exist_ok=True)
            write(FOUT / 'status.json', {'time': now(), 'status': 'running', 'mode': mode,
                'repeat': repeat, 'case': [index, seed, arm],
                'completed': sum(destination(mode, repeat, *t).exists() for t in todo), 'total_in_process': len(todo)})
            sample_info = info['samples'][str(index)]
            payload = torch.load(verify(sample_info['payload']), map_location='cpu', weights_only=True)
            ss = info['ss'][f'{index}/{seed}/{arm}']
            with np.load(verify(ss['coords'])) as data:
                coords = canonical_coords(data['coords'])
            noise_info = info['noises'][f'{index}/{seed}']
            master = torch.load(verify(noise_info['file']), map_location='cpu', weights_only=True)
            if tensor_meta(master) != noise_info['tensor']:
                raise ValueError('Changed paired master noise')
            initial = sparse_noise(coords, master.to('cuda:0'))
            initial_data = sparse_payload(initial)
            initial_file = save_tensor_file(dest.parent / 'initial.pt', initial_data)
            initial_sha = {k: tensor_meta(v) for k, v in initial_data.items()}
            latent, calls = sample_mesh_latent(runtime, payload['sample'], initial)
            if (calls['positive_calls'], calls['negative_calls']) != (25, 19):
                raise ValueError('Incomplete SLat sampling')
            latent_data = sparse_payload(latent)
            latent_file = save_tensor_file(dest.parent / 'latent.pt', latent_data)
            latent_sha = {k: tensor_meta(v) for k, v in latent_data.items()}
            mesh = runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError('Empty/nonfinite mesh; stop without dropping the target')
            mesh_path = dest.parent / 'mesh.ply'
            mesh.export(mesh_path)
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            target = trimesh.load(verify(sample_info['gt']), force='mesh', process=False)
            mesh_sha = {k: tensor_meta(torch.from_numpy(v.copy())) for k, v in {
                'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.faces)}.items()}
            row = {'time': now(), 'protocol': binding(FOUT / 'protocol.json'), 'mode': mode, 'repeat': repeat,
                'index': index, 'seed': seed, 'arm': arm, 'uid': payload['uid'], 'object_uid': payload['object_uid'],
                'segment': binding(segment_path), 'weights': weights, 'input_payload': sample_info['payload'],
                'source_ss': ss['source'], 'master_noise': noise_info['file'], 'initial': initial_file,
                'initial_sha': initial_sha, 'latent': latent_file, 'latent_sha': latent_sha,
                'mesh': binding(mesh_path), 'mesh_sha': mesh_sha, 'gt': sample_info['gt'], 'calls': calls,
                'surface': surface_metrics(mesh, target), 'structure': structure_metrics(mesh),
                'bbox_min': np.asarray(mesh.vertices).min(0).tolist(), 'bbox_max': np.asarray(mesh.vertices).max(0).tolist(),
                'success': True, 'fitted_transform': False, 'policy_so_far': fixed.manifest(),
                'elapsed_seconds': time.monotonic()-begin,
                'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/(1 << 20)}
            write(dest, row)
            print('FAIR_MESH', mode, repeat, index, seed, arm, row['surface']['chamfer_l1'], flush=True)
            del initial, latent, mesh, target, master, initial_data, latent_data
    if any(p._version != versions[k][n] or p.grad is not None for k in weights for n, p in runtime[k].named_parameters()):
        raise ValueError('Frozen model changed during inference')
    write(FOUT / mode / str(repeat) / 'complete.json', {'time': now(),
        'protocol': binding(FOUT / 'protocol.json'), 'mode': mode, 'repeat': repeat,
        'records': [binding(destination(mode, repeat, *t)) for t in todo]})


def compare(a, b):
    return {'initial_exact': a['initial_sha'] == b['initial_sha'],
        'latent_exact': a['latent_sha'] == b['latent_sha'], 'mesh_exact': a['mesh_sha'] == b['mesh_sha'],
        'mesh_file_exact': a['mesh']['sha256'] == b['mesh']['sha256'],
        'metric_exact': a['surface'] == b['surface'],
        'cd_delta': b['surface']['chamfer_l1']-a['surface']['chamfer_l1']}


def summarize(mode):
    rows = [[check_row(destination(mode, r, *t)) for t in tasks(mode, r)] for r in [0, 1]]
    segments = [read(verify(v['segment'])) for rr in rows for v in rr]
    if any(s['environment'] != segments[0]['environment'] or s['patch'] != segments[0]['patch'] for s in segments):
        raise ValueError('Compared runs used different environments or mesh conversion patches')
    if mode == 'smoke':
        comparisons = [compare(rows[0][0], rows[1][0])]
    else:
        by_key = [{(v['index'], v['seed'], v['arm']): v for v in rr} for rr in rows]
        comparisons = [{'index': i, 'seed': s, 'arm': a, **compare(by_key[0][i,s,a], by_key[1][i,s,a])}
            for i in INDICES for s in SEEDS for a in ARMS]
        # The original shelf's uniform/balanced supports coincide for all seeds.
        comparisons.extend({'repeat': r, 'index': 2, 'seed': s, 'same_support_arms': True,
            **compare(by_key[r][2,s,'uniform'], by_key[r][2,s,'balanced'])} for r in [0,1] for s in SEEDS)
    passed = all(all(c[k] for k in ['initial_exact','latent_exact','mesh_exact','mesh_file_exact','metric_exact']) for c in comparisons)
    report = {'time': now(), 'protocol': binding(FOUT / 'protocol.json'), 'mode': mode,
        'passed_reproducibility': passed, 'comparisons': comparisons, 'records': rows,
        'scope': 'same four selected training objects; only numerical reproducibility is a pass gate'}
    if mode == 'run':
        report['means'] = {a: {m: float(np.mean([v['surface'][m] for v in rows[0] if v['arm']==a]))
            for m in ['chamfer_l1', 'fscore_0p02']} for a in ARMS}
        report['paired_objects'] = []
        for control in ['uniform', 'appearance_control']:
            for i in INDICES:
                deltas = [by_key[0][i,s,'balanced']['surface']['chamfer_l1']-
                    by_key[0][i,s,control]['surface']['chamfer_l1'] for s in SEEDS]
                report['paired_objects'].append({'index':i, 'candidate':'balanced', 'control':control,
                    'cd_delta_by_seed':deltas, 'mean_cd_delta':float(np.mean(deltas))})
        if passed:
            from .fair_visuals import galleries
            report['galleries'] = galleries(rows)
    write(FOUT / f'{mode}_summary.json', report)
    if mode == 'run':
        lines = ['# HSSD4 固定内核公平 mesh 复核', '', f'完成时间：{now()}。', '',
            f'72 个 mesh；跨进程/顺序复现检查：{passed}。四个预先选定的训练物体、三个采样 seed、两个执行顺序。', '',
            '三组重新生成，权重、实际坐标噪声、精度和内核选择规则相同。GT 与预测处于原生坐标系，无 ICP 或旋转/平移/尺度拟合。', '',
            '| 方法 | CD ↓ | F@0.02 ↑ |', '|---|---:|---:|']
        lines += [f"| {a} | {report['means'][a]['chamfer_l1']:.9f} | {report['means'][a]['fscore_0p02']:.6f} |" for a in ARMS]
        lines += ['', '表中为执行顺序 0 的均值；顺序 1 用于独立复现，不能作为额外独立样本。', '',
            '| index | balanced − uniform CD | balanced − appearance CD |', '|---|---:|---:|']
        for i in INDICES:
            dd = {v['control']: v['mean_cd_delta'] for v in report['paired_objects'] if v['index']==i}
            lines.append(f"| {i} | {dd['uniform']:+.9f} | {dd['appearance_control']:+.9f} |")
        lines += ['', '这是四个训练诊断物体的复核，不能替代独立场景验证或宣称论文贡献成立。',
            '若复现检查未通过，不解释表中的细小方法差异。若仍未超过同容量外观对照，停止当前配置的扩量和长训，先修改几何条件注入的假设。', '',
            '[完整结果](run_summary.json)；[固定协议](protocol.json)；[同相机 mesh 图库](figures/)。', '']
        (FOUT / 'REPORT.md').write_text('\n'.join(lines))
    write(FOUT / 'status.json', {'time': now(), 'status': mode+'_complete_review_required' if passed else 'reproducibility_failed',
        'mesh_count':sum(map(len,rows)), 'passed_reproducibility':passed})
    if not passed:
        raise RuntimeError('Identical-input reproducibility gate failed; do not interpret small quality differences')
    return report


def launch(mode):
    (FOUT / 'inputs').mkdir(parents=True, exist_ok=True)
    prepare()
    if mode == 'run':
        gate = read(FOUT / 'smoke_summary.json')
        if gate['protocol'] != binding(FOUT / 'protocol.json') or not gate['passed_reproducibility']:
            raise ValueError('Both independent complete-path smoke runs must pass first')
    # Run serially on the one requested GPU; each worker is a new process.
    for repeat in [0, 1]:
        log = FOUT / f'{mode}_{repeat}_{time.time_ns()}.log'
        with log.open('w') as stream:
            process = subprocess.Popen([PYTHON, '-u', '-m', MODULE+'.fair', 'worker',
                '--mode', mode, '--repeat', str(repeat)], stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
            try:
                code = process.wait(timeout=600 if mode=='smoke' else 14400)
                if code: raise RuntimeError(f'Fair {mode} worker {repeat} exited {code}; see {log}')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
    summarize(mode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['smoke','run','worker','summary'])
    p.add_argument('--mode', choices=['smoke','run'], default='run')
    p.add_argument('--repeat', type=int, choices=[0,1], default=0)
    args = p.parse_args()
    def interrupted(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.action=='worker': worker(args.mode,args.repeat)
    elif args.action=='summary': summarize(args.mode)
    else: launch(args.action)
