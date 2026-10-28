"""Final fourth-round review, visual comparisons, and append-only audit notes."""
import argparse
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from visual_evidence_probe.experiments.local_repair import config


def perfect_controls(context):
    from visual_evidence_probe.experiments.local_repair.runner import load_gt, raster
    from visual_evidence_probe.analysis_scripts.local_repair_cropped import observation
    from visual_evidence_probe.experiments.local_repair.photometric import prepare_images, propose
    out = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
    rows = []
    for index in config.PARAMS['indices']:
        rec, k, w, images, masks = observation(index)
        gt = load_gt(index)
        pred = raster(gt.vertices, gt.faces, k, w, context, config.PARAMS['side'])
        _, gray, _, eroded = prepare_images(images, masks, config.PARAMS['side'])
        p = propose(pred, gray, eroded, k, w)
        initial_supported = p['support'] > 0
        valid = p['valid']
        row = dict(index=index, uid=rec['uid'], gt_geometry=True, input_only_matching=True,
            supported_pixels=int(initial_supported.sum()), false_proposal_pixels=int(valid.sum()),
            false_proposal_rate=float(valid.sum() / max(initial_supported.sum(), 1)),
            mean_abs_false_delta=float(np.abs(p['delta'][valid]).mean()) if valid.any() else None,
            warning='Proposal false alarms only. No large GT-mesh editing is performed; this is not accepted-edit specificity.')
        config.write(out / 'perfect_geometry_controls' / f'{index:03d}.json', row)
        rows.append(row)
        print(dict(stage='perfect_geometry_matching_control', **row), flush=True)
        del gt, pred; gc.collect(); torch.cuda.empty_cache()
    config.write(out / 'perfect_geometry_controls/summary.json', dict(time=config.now(), records=rows,
        total_false_proposal_pixels=sum(r['false_proposal_pixels'] for r in rows),
        total_supported_pixels=sum(r['supported_pixels'] for r in rows)))


def shaded(pred, view):
    xyz = pred['xyz'][view].cpu().numpy()
    mask = pred['mask'][view].cpu().numpy()
    dy, dx = np.gradient(xyz, axis=(0, 1))
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.e-8)
    # Double-sided neutral Lambert visualization; depth buffers/cameras are identical across rows.
    light = np.array([.35, -.45, .8]); light /= np.linalg.norm(light)
    brightness = .30 + .65 * np.abs(normal @ light)
    image = np.ones((*mask.shape, 3), np.float32)
    image[mask] = brightness[mask, None] * np.array([.65, .74, .81])
    return image


def figures(context):
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visual_evidence_probe.experiments.local_repair.runner import load_gt, raster
    out = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
    for index in config.PARAMS['indices']:
        gt = load_gt(index)
        rec = config.read(config.BASE / 'objects' / f'{index:03d}' / 'observations.json')
        k = torch.tensor(rec['intrinsics'], device='cuda:0', dtype=torch.float32)
        w = torch.tensor(rec['extrinsics'], device='cuda:0', dtype=torch.float32)
        gt_image = raster(gt.vertices, gt.faces, k, w, context, 256)
        for seed in config.PARAMS['seeds']:
            root = out / 'objects' / f'{index:03d}' / str(seed)
            if not (root / 'result.json').exists():
                continue
            record = config.read(root / 'result.json')
            rows = [('GT', None), ('Frozen30K + same global fit', config.BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.obj'),
                ('Automatic accepted', root / 'automatic.obj'), ('Automatic full proposal', root / 'automatic_proposal.ply'),
                ('Random full proposal', root / 'random_proposal.ply'),
                ('GT-region full proposal (diagnostic)', root / 'oracle_region_proposal.ply'),
                ('GT-direction full proposal (diagnostic)', root / 'oracle_direction_proposal.ply')]
            fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.5 * len(rows)))
            for row, (name, path) in enumerate(rows):
                if path is None:
                    pred = gt_image
                else:
                    mesh = trimesh.load(path, process=False, force='mesh', skip_materials=True)
                    pred = raster(mesh.vertices, mesh.faces, k, w, context, 256)
                for view in range(3):
                    axes[row, view].imshow(shaded(pred, view)); axes[row, view].axis('off')
                    if view == 0:
                        axes[row, view].set_title(name, fontsize=9, loc='left')
            fig.suptitle(f'{rec["uid"]}, seed{seed}; same frame/cameras; diagnostic oracle rows use GT', fontsize=10)
            fig.tight_layout()
            dest = out / 'figures' / f'{index:03d}_{seed}.jpg'; dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=130); plt.close(fig)
        del gt, gt_image; gc.collect(); torch.cuda.empty_cache()


def finalize():
    from visual_evidence_probe.experiments.local_repair import report
    strict = config.STORE / 'outputs/local_geometry_repair_20261026'
    crop = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
    for out in (strict, crop):
        report.OUT = out; report.main()
    a, b = config.read(strict / 'summary.json'), config.read(crop / 'summary.json')
    control = config.read(crop / 'perfect_geometry_controls/summary.json')
    lines = ['# 第四次验证结论：局部修复闭环', '', f'时间：{config.now()}', '',
        '已完成原图严格版和输入裁剪补充版，各8对象×2seed×4修改臂。',
        '所有结果复用第三轮已检查的对象，属于探索性验证，无模型训练。', '',
        '|版本/方法|接受修改|有修改约束|接受后CD改善|完整proposal的CD改善|',
        '|---|---:|---:|---:|---:|']
    for title, s in [('原图严格', a), ('输入裁剪', b)]:
        for m, values in s['methods'].items():
            cd = values['accepted']['surface/chamfer_l1']['mean_improvement']
            raw = values['proposed']['surface/chamfer_l1']['mean_improvement']
            lines.append(f'|{title}/{m}|{values["accepted_cases"]}/16|{values["nonzero_proposal_cases"]}/16|{cd:+.6f}|{raw:+.6f}|')
    lines += ['', '正值为改善。oracle_region使用GT选区，oracle_direction还使用GT方向，均不能算部署方法。',
        f'正确GT几何上的输入匹配仍产生 {control["total_false_proposal_pixels"]} 个非零候选，',
        f'支持像素合计 {control["total_supported_pixels"]}。这里只测候选误报，未测对GT mesh最终接受修改的比例。', '',
        '## 解释边界', '',
        '- 残差定位与修复方向是两个不同问题；定位指标更好不保证局部更新有效。',
        '- 无proposal、无第4视图支持、拒绝修改的案例全部保留；不能把零修改说成修复成功。',
        '- 第4张图仅未参与局部深度搜索，已经参与原生成和全局拟合。',
        '- 裁剪版在首例严格版零proposal后登记；两版分开报告，原版结果保留。',
        '- SLat字段代理优化与实际mesh端点评价另见local_slat_repair_20261026。', '',
        '[严格版报告](outputs/local_geometry_repair_20261026/REPORT.md)、',
        '[裁剪版报告](outputs/local_geometry_repair_cropped_20261026/REPORT.md)、',
        '[对比图目录](outputs/local_geometry_repair_cropped_20261026/figures/)。', '']
    dest = config.STORE / '第四次验证结论.md'
    dest.write_text('\n'.join(lines))
    # Append once; never replace the earlier audit.
    audit = config.STORE / '审计.md'
    marker = '# 4. 局部几何修复闭环（2026-10-26）'
    text = audit.read_text()
    if marker not in text:
        with audit.open('a') as stream:
            stream.write('\n\n' + marker + '\n\n' + '\n'.join(lines[2:]) + '\n')
    print(dict(conclusion=str(dest), audit_appended=True), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['controls', 'figures', 'finalize'])
    p.add_argument('--gpu', default='0'); args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu; config.setup()
    if args.action == 'finalize':
        finalize(); return
    from visual_evidence_probe.runtime import setup
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=setup())
    if args.action == 'controls': perfect_controls(context)
    else: figures(context)


if __name__ == '__main__': main()
