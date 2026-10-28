"""Equal final postprocessing: separate registration errors from shape errors.

Reuse the frozen two-mask fitter without changing its source or any old result.
"""
import gc
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from visual_evidence_probe.common import binding, read, write, verify, lock_json, now, ROOT
from visual_evidence_probe.experiments.input_consistency import config, refine

OUT = config.OUT / 'final_output_alignment_control'
KEYS = ('raw42', 'raw43', 'aligned42', 'aligned43')


def folder(index):
    return OUT / 'objects' / f'{index:03d}'


def emit(stage, **kwargs):
    value = dict(time=now(), stage=stage, **kwargs)
    write(OUT / 'status.json', value)
    print(value, flush=True)


def observations(index):
    rec, k, w, masks = config.observations(index)
    chosen = next(r['selected'] for r in read(config.OUT / 'raw_decisions.json')['records'] if r['index'] == index)
    donors = {}
    for seed in config.SEEDS:
        donors[f'raw{seed}'] = binding(config.BASE / 'objects' / f'{index:03d}' / f'results/donor_{chosen}6/{seed}.obj')
        donors[f'aligned{seed}'] = binding(config.folder(index) / f'resampled/{seed}.obj')
    return {**rec, 'donors': donors}, k, w, masks


def prepare():
    config.verify_protocol()
    protocol = dict(time_scope='2026-10-25 UTC; exploratory control added after primary results',
        reason='same final silhouette fitting for real2 and pseudo-view outputs; remove remaining global-registration confound',
        sources=[binding(__file__), binding(Path(__file__).with_name('OUTPUT_ALIGNMENT_PROTOCOL.md'))],
        parent=binding(config.OUT / 'protocol.json'), params=config.PARAMS,
        indices=list(config.INDICES), keys=list(KEYS), uses_gt_for_fitting=False,
        source_identity='same raw preselected A/B/A/A; no GT choice',
        prior_results_seen=True, unchanged_fitter=True)
    lock_json(OUT / 'protocol.json', protocol)
    for i in config.INDICES:
        rec, _, _, _ = observations(i)
        lock_json(folder(i) / 'inputs.json', rec)


def score(index, key):
    import trimesh
    from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics, structure_metrics
    path = folder(index) / 'refined' / f'{key}.metrics.json'
    if path.exists():
        verify(read(path)['mesh'])
        return read(path)
    fitted = read(folder(index) / 'refined' / f'{key}.json')
    target = next(r for r in read(config.OUT / 'evaluation_targets.json')['records'] if r['index'] == index)
    mesh = trimesh.load(verify(fitted['mesh']), force='mesh', process=False)
    gt = trimesh.load(verify(target['gt']), force='mesh', process=False)
    result = dict(index=index, key=key, seed=int(key[-2:]), success=True,
        surface=surface_metrics(mesh, gt, count=50000, seed=20261025),
        structure=structure_metrics(mesh), occupancy=None, mesh=fitted['mesh'],
        gt=target['gt'], uses_gt_for_inference=False, fit=binding(folder(index) / 'refined' / f'{key}.json'))
    write(path, result)
    return result


def gallery(index, context):
    import trimesh
    from visual_evidence_probe.render import render
    _, k, w, _ = config.observations(index)
    target = next(r for r in read(config.OUT / 'evaluation_targets.json')['records'] if r['index'] == index)
    rows = [('GT', target['gt'], None)]
    for seed, key in [(42, 'a'), (43, 'b')]:
        r = read(config.folder(index) / f'refined/{key}.metrics.json')
        rows.append((f'real2 + final fit / {seed}', r['mesh'], r))
    for prefix in ('raw', 'aligned'):
        for seed in config.SEEDS:
            r = read(folder(index) / f'refined/{prefix}{seed}.metrics.json')
            rows.append((f'{prefix} extra + final fit / {seed}', r['mesh'], r))
    side, label = 220, 250
    canvas = Image.new('RGB', (label+side*4, side*len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, source, metrics) in enumerate(rows):
        draw.text((8, row*side+55), name, fill='black')
        if metrics:
            draw.text((8, row*side+80), f"CD {metrics['surface']['chamfer_l1']:.5f}", fill='black')
            draw.text((8, row*side+100), f"F@.02 {metrics['surface']['fscore_0p02']:.4f}", fill='black')
        mesh = trimesh.load(verify(source), force='mesh', process=False)
        images, _ = render(mesh, k[[0, 1, 3, 6]], w[[0, 1, 3, 6]], side=side, context=context)
        for j, image in enumerate(images):
            canvas.paste(image, (label+j*side, row*side))
    canvas.save(folder(index) / 'equal_final_fit_comparison.jpg', quality=93)


def report(context):
    from visual_evidence_probe.experiments.input_consistency.report import summarize
    baseline = [read(config.BASE / 'objects' / f'{i:03d}' / f'results/real2/{s}.json')
                for i in config.INDICES for s in config.SEEDS]
    direct = [read(config.folder(i) / f'refined/{k}.metrics.json') for i in config.INDICES for k in ('a', 'b')]
    records = {(i, key): score(i, key) for i in config.INDICES for key in KEYS}
    groups = dict(real2_unfitted=baseline, real2_final_fit=direct)
    groups.update({f'{prefix}_extra_final_fit': [records[i, f'{prefix}{s}'] for i in config.INDICES for s in config.SEEDS]
                   for prefix in ('raw', 'aligned')})
    summaries = {name: summarize(r, direct) for name, r in groups.items()}
    fits = [read(folder(i) / f'refined/{key}.json') for i in config.INDICES for key in KEYS]
    result = dict(time=now(), objects=4, refined_outputs=16, summaries=summaries,
        delta_reference='real2 with the SAME final similarity fit; summarize internal gain_over_real2 field uses this reference',
        mean_final_fit_seconds=float(np.mean([r['elapsed_seconds'] for r in fits])),
        all_successful=True, uses_gt_for_fitting=False,
        scope='post-hoc same-scene diagnostic, not independent validation')
    write(OUT / 'summary.json', result)
    lines = ['# 相同最终轮廓修正的控制实验', '', f"完成：{result['time']}", '',
        '原始两图、原始供图回灌、修正供图回灌的最终mesh，均接受相同的两输入mask相似变换拟合。',
        '不改变局部形状/拓扑、不用GT挑变换；GT只用于封存变换后的评估。',
        '该控制在第二轮主要结果之后追加，仍为同一批已检查样本的事后诊断。', '',
        '|方案|CD↓|F@.02↑|CD优/差/同于同样修正的两图基线|', '|---|---:|---:|---:|']
    for name, v in summaries.items():
        lines.append(f"|{name}|{v['cd']['mean']:.8f}|{v['f02']['mean']:.6f}|{v['cd']['wins']}/{v['cd']['losses']}/{v['cd']['ties']}|")
    lines += ['', f"额外16个mesh拟合平均每个 {result['mean_final_fit_seconds']:.2f} 秒；三维指标沿用50000点、seed20261025。", '',
        '如果回灌结果在同样最终修正后仍更差，说明当前结果的差异不只是这些有界全局变换能消除的对齐误差。',
        '这仍不能单独确立来源相关性机制，也不能否认其他来源、更高质量图像或其他模型可能有效。', '',
        '[数值](summary.json)；[控制方案](../../../analysis_scripts/OUTPUT_ALIGNMENT_PROTOCOL.md)。', '']
    for i in config.INDICES:
        lines.append(f'- 对象{i:03d}：[相同最终修正对照图](objects/{i:03d}/equal_final_fit_comparison.jpg)。')
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    for i in config.INDICES:
        gallery(i, context)
    return result


def run():
    import nvdiffrast.torch as dr
    from visual_evidence_probe.runtime import setup
    prepare()
    device = setup()
    context = dr.RasterizeCudaContext(device=device)
    start = time.monotonic()
    with patch.object(refine, 'folder', folder), patch.object(refine, 'observations', observations), patch.object(refine, 'emit', emit):
        for i in config.INDICES:
            for key in KEYS:
                refine.fit(i, key, context, device)
                gc.collect()
    # All 16 choices are sealed before target geometry is loaded.
    lock_json(OUT / 'sealed_fit_choices.json', {'records': [binding(folder(i) / f'refined/{key}.json')
              for i in config.INDICES for key in KEYS], 'uses_gt': False})
    report(context)
    emit('complete', refined_meshes=16, elapsed_seconds=time.monotonic()-start)


def main():
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.6')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'outputs/torch_extensions'))
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    run()


if __name__ == '__main__':
    main()
