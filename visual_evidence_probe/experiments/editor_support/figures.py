"""Saved-mesh figures for round five; no model or decoder is loaded.

python -B -m visual_evidence_probe.experiments.editor_support.figures --phase all --gpu 0
Only cases with their completion receipts are included. Every row uses the
saved world-space mesh and exactly the same posed input cameras; no additional
alignment, per-row scale, or per-row framing is applied.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from .config import BASE, OUT, PARAMS, folder, now, read, setup, write


FIGURES = OUT / 'figures'


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'figure.facecolor': 'white'})
    return plt


def _cameras(index, device):
    # The optimizer used the crop-resolution-control cameras, not the old
    # uncropped round-three observations. Reusing these is essential for parity.
    path = BASE / 'input_crop_resolution_control/objects' / f'{index:03d}' / 'observations.json'
    record = read(path)
    return record, torch.tensor(record['intrinsics'], dtype=torch.float32, device=device), \
        torch.tensor(record['extrinsics'], dtype=torch.float32, device=device), path


@torch.no_grad()
def _render(path, k, w, context, side):
    import trimesh
    from .objective import raster_tensor
    mesh = trimesh.load(path, process=False, force='mesh', skip_materials=True)
    vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device=k.device)
    faces = torch.as_tensor(np.asarray(mesh.faces).copy(), dtype=torch.int32, device=k.device)
    pred = raster_tensor(vertices, faces, k, w, context, side=side)
    # Retain just the small buffers needed by the figures, not face/mesh arrays.
    result = {name: pred[name].cpu() for name in ('depth', 'mask', 'xyz')}
    receipt = dict(path=str(path), physical_path=str(Path(path).resolve()),
                   vertices=len(vertices), faces=len(faces), world_bounds=np.asarray(mesh.bounds).tolist(),
                   hit_pixels=pred['mask'].sum((1, 2)).cpu().tolist())
    del mesh, vertices, faces, pred
    return result, receipt


def _images(pred, views):
    from visual_evidence_probe.analysis_scripts.local_repair_review import shaded
    # Same neutral, double-sided Lambert visualization as the fourth-round
    # figures. Illumination and display limits remain constant across all rows.
    return [shaded(pred, view) for view in views]


def _grid(rows, views, title, subtitle, dest, footnote, view_labels=None):
    plt = _plotting()
    fig, axes = plt.subplots(len(rows), len(views), figsize=(10.4, 2.05 * len(rows) + 1.05), squeeze=False)
    for row, record in enumerate(rows):
        for column, view in enumerate(views):
            axis = axes[row, column]
            axis.imshow(record['images'][column], interpolation='nearest', vmin=0, vmax=1)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.text(-.055, .5, record['label'], transform=axis.transAxes,
                          ha='right', va='center', fontsize=8.4, linespacing=1.35)
            if row == 0:
                axis.set_title(view_labels[column] if view_labels else f'Input view {view}')
    fig.suptitle(title + '\n' + subtitle, fontsize=10.4, y=.985, linespacing=1.5)
    fig.text(.5, .012, footnote, ha='center', va='bottom', fontsize=8)
    fig.subplots_adjust(left=.255, right=.995, bottom=.035, top=1 - .85 / fig.get_figheight(),
                        wspace=.025, hspace=.035)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=125, pil_kwargs={'quality': 92})
    plt.close(fig)


def _error_grid(rows, target, views, title, dest):
    plt = _plotting()
    target_depth, domain = target['depth'].numpy(), target['mask'].numpy()
    initial = rows[0]['pred']
    initial_error = np.where(initial['mask'].numpy(), np.abs(initial['depth'].numpy() - target_depth), .1)
    cap = max(float(np.quantile(initial_error[domain], .99)), .0005) if domain.any() else .001
    fig, axes = plt.subplots(len(rows), len(views), figsize=(10.4, 1.9 * len(rows) + 1), squeeze=False)
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('white')
    last = None
    for row, record in enumerate(rows):
        pred = record['pred']
        # Fixed target domain; missing predicted support is charged .1 rather
        # than silently disappearing from the visualization or denominator.
        error = np.where(pred['mask'].numpy(), np.minimum(np.abs(pred['depth'].numpy() - target_depth), .1), .1)
        for column, view in enumerate(views):
            axis = axes[row, column]
            last = axis.imshow(np.ma.masked_where(~domain[view], error[view]), cmap=cmap, vmin=0, vmax=cap,
                               interpolation='nearest')
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_visible(False)
            if column == 0:
                mae = float(error[domain].mean()) if domain.any() else float('nan')
                axis.text(-.055, .5, record['label'] + f'\nfixed-domain MAE={mae:.6f}',
                          transform=axis.transAxes, ha='right', va='center', fontsize=8)
            if row == 0: axis.set_title(f'Input view {view}')
    fig.suptitle(title + '\nDepth error to artificial reachable target; scan GT is not used', fontsize=10, y=.985)
    fig.subplots_adjust(left=.27, right=.91, bottom=.045, top=.92, hspace=.04, wspace=.025)
    cax = fig.add_axes([.935, .23, .015, .5]); fig.colorbar(last, cax=cax, label='Absolute camera-z error')
    fig.text(.5, .012, f'One fixed target mask and color scale for all rows; saturated above {cap:.6f}. Missing rays pay 0.1.',
             ha='center', va='bottom', fontsize=8)
    fig.savefig(dest, dpi=125, pil_kwargs={'quality': 92}); plt.close(fig)
    return dict(color_scale_max=cap, missing_ray_penalty=.1, domain='fixed reachable-target mask')


def recovery_case(index, seed, context, device, side):
    path = folder(index, seed, 'recovery')
    results = {}
    for objective in ('proxy', 'mesh'):
        result_path = path / objective / 'result.json'
        if not result_path.exists(): return None
        result = read(result_path)
        if result.get('status') != 'complete': return None
        results[objective] = result
    rec, k, w, camera_path = _cameras(index, device)
    views = [0, 1, 2]
    sources = [('Frozen30K\n+ same global fit', BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.obj'),
               ('Reachable target\n(artificial latent)', path / 'reachable_target.ply')]
    for objective in ('proxy', 'mesh'):
        result = results[objective]
        name = 'Corner-SDF proxy' if objective == 'proxy' else 'Full-mesh loss'
        sources.extend([(f'{name}\nfull proposal, step {result["steps"]}', path / objective / 'proposal.ply'),
            (f'{name}\nselected step {result["best_step"]}' +
             (' (NO EDIT)' if not result.get('actual_edit', False) else '') +
             f'\ndepth reduction {result["depth_error_reduction"]:.1%}', path / objective / 'accepted.obj')])
    rows, receipts = [], []
    cache = {}
    for label, source in sources:
        key = source.resolve()
        if key not in cache: cache[key] = _render(source, k, w, context, side)
        pred, receipt = cache[key]
        rows.append(dict(label=label, pred=pred, images=_images(pred, views)))
        receipts.append(dict(label=label, **receipt))
    dest = FIGURES / 'recovery' / f'{index:03d}_{seed}.jpg'
    title = f'Known-latent recovery: {rec["uid"]}, seed {seed}'
    _grid(rows, views, title, 'Artificial reachable latent target; NOT scan GT | Same frame, camera K/W and lighting', dest,
          'All four input views supervise this development diagnostic. Proposal and selected output are shown separately.')
    error_dest = dest.with_name(dest.stem + '_depth_error.jpg')
    errors = _error_grid([rows[i] for i in (0, 2, 3, 4, 5)], rows[1]['pred'], views, title, error_dest)
    result = dict(index=index, seed=seed, uid=rec['uid'], figure=str(dest), error_figure=str(error_dest),
        cameras=str(camera_path), views=views, side=side, same_world_frame=True, new_alignment=False,
        artificial_reachable_target=True, scan_gt_used=False, rows=receipts, error_visualization=errors,
        depth_error_reduction={name: value['depth_error_reduction'] for name, value in results.items()})
    print(dict(stage='recovery_figure', index=index, seed=seed, figure=str(dest)), flush=True)
    return result


def _accepted_label(name, result):
    if not result.get('accepted_edit', False):
        return f'{name}\nactual accepted: NO EDIT\n= original baseline'
    return f'{name}\nactual accepted step {result.get("best_step", "?")}'


def factorial_case(index, seed, context, device, side):
    path = folder(index, seed)
    if not (path / 'complete.json').exists(): return None
    complete = read(path / 'complete.json')
    results = {}
    for name in ('original_proxy', 'original_mesh', 'expanded_proxy', 'expanded_mesh'):
        result_path = path / name / 'result.json'
        if not result_path.exists(): return None
        results[name] = read(result_path)
        if results[name].get('status') == 'failed': return None
    pool = read(path / 'input_pool.json')
    rec, k, w, camera_path = _cameras(index, device)
    heldout = pool.get('heldout_views', [])
    train = pool.get('train_views', [])
    views = (train[:2] + heldout[:1]) if heldout and len(train) >= 2 else [0, 1, 2]
    view_labels = [f'Input view {view}' + (' | edit holdout' if view in heldout else '') for view in views]
    init_status = results['expanded_mesh'].get('initialization_status', results['expanded_mesh'].get('status', 'unknown'))
    sources = [('Scan GT canonical\n(diagnostic reference only)', BASE / 'objects' / f'{index:03d}' / 'target_canonical.ply'),
               ('Original full mesh\n(before local edit)', path / 'original.obj'),
               ('Original SS + proxy\nfull proposal', path / 'original_proxy/proposal.ply'),
               ('Original SS + mesh loss\nfull proposal', path / 'original_mesh/proposal.ply'),
               (f'Expanded SS: step 0\ninitialization: {init_status}', path / 'expanded_mesh/step0.ply'),
               ('Expanded SS + proxy\nfull proposal', path / 'expanded_proxy/proposal.ply'),
               ('Expanded SS + mesh loss\nfull proposal', path / 'expanded_mesh/proposal.ply'),
               (_accepted_label('Original SS + mesh loss', results['original_mesh']), path / 'original_mesh/accepted.obj'),
               (_accepted_label('Expanded SS + mesh loss', results['expanded_mesh']), path / 'expanded_mesh/accepted.obj')]
    rows, receipts, cache = [], [], {}
    base_pred = None
    alignment = {}
    for row, (label, source) in enumerate(sources):
        key = source.resolve()
        if key not in cache: cache[key] = _render(source, k, w, context, side)
        pred, receipt = cache[key]
        if row == 1: base_pred = pred
        if row in (4, 7, 8):
            common = pred['mask'] & base_pred['mask']
            error = (pred['depth'] - base_pred['depth']).abs()[common]
            union = pred['mask'] | base_pred['mask']
            alignment[str(row)] = dict(depth_mae_to_original=float(error.mean()) if error.numel() else None,
                mask_iou_to_original=((pred['mask'] & base_pred['mask']).sum((1, 2)) /
                    union.sum((1, 2)).clamp_min(1)).tolist())
        rows.append(dict(label=label, images=_images(pred, views)))
        receipts.append(dict(label=label, **receipt))
    dest = FIGURES / 'factorial' / f'{index:03d}_{seed}.jpg'
    subtitle = f'Same world frame, camera K/W and lighting | constraints={complete["constraint_count"]} | {pool["status"]}'
    _grid(rows, views, f'Input-only local editing: {rec["uid"]}, seed {seed}', subtitle, dest,
          'Scan GT is shown only for diagnosis. Proposals can be rejected; actual accepted outputs are the final two rows.', view_labels)
    result = dict(index=index, seed=seed, uid=rec['uid'], figure=str(dest), complete_receipt=str(path / 'complete.json'),
        cameras=str(camera_path), views=views, heldout_views=heldout, train_views=train, side=side,
        same_world_frame=True, new_alignment=False, gt_row='diagnostic reference only', rows=receipts,
        initialization_status=init_status, input_status=pool['status'], constraint_count=complete['constraint_count'],
        accepted={name: bool(value.get('accepted_edit', False)) for name, value in results.items()},
        render_consistency=alignment)
    print(dict(stage='factorial_figure', index=index, seed=seed, figure=str(dest)), flush=True)
    return result


def recovery_summary():
    plt = _plotting()
    records = []
    for index in PARAMS['recovery_indices']:
        for seed in PARAMS['seeds']:
            pair = {}
            for objective in ('proxy', 'mesh'):
                path = folder(index, seed, 'recovery') / objective / 'result.json'
                if path.exists() and read(path).get('status') == 'complete': pair[objective] = read(path)
            if len(pair) == 2: records.append((index, seed, pair))
    if not records: return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(records))
    for offset, objective, color, label in ((-.18, 'proxy', '#90959c', 'Corner-SDF proxy'),
                                            (.18, 'mesh', '#157c9b', 'Full-mesh loss')):
        values = [pair[objective]['depth_error_reduction'] * 100 for _, _, pair in records]
        axes[0].bar(x + offset, values, width=.34, color=color, label=label)
        for position, value in zip(x + offset, values):
            axes[0].text(position, value + 1, f'{value:.1f}%', ha='center', fontsize=8)
        for index, seed, pair in records:
            r = pair[objective]
            initial = max(r['initial_depth_loss'], 1.e-12)
            axes[1].plot([point['step'] for point in r['snapshots']],
                [point['local_depth'] / initial for point in r['snapshots']],
                color=color, alpha=.6, marker='o', markersize=3, linewidth=1.2,
                linestyle='--' if seed == 43 else '-')
    axes[0].axhline(50, color='#b24a32', linestyle=':', label='50% development threshold')
    axes[0].set_xticks(x, [f'{index:03d} / seed {seed}' for index, seed, _ in records])
    axes[0].set_ylabel('Selected depth-error reduction (%)'); axes[0].set_ylim(-5, 100)
    axes[0].legend(loc='upper right', fontsize=8); axes[0].grid(axis='y', alpha=.2)
    axes[1].set_xlabel('Optimization step'); axes[1].set_ylabel('Snapshot depth loss / initial depth loss')
    axes[1].set_ylim(bottom=0); axes[1].grid(alpha=.2)
    axes[1].set_title('All four cases; dashed=seed43, solid=seed42', fontsize=9)
    fig.suptitle(f'Known-latent recovery ({len(records)}/4 completed cases): artificial reachable targets, NOT scan GT', fontsize=11)
    fig.text(.5, .025, 'A supervised development diagnostic on targets produced by the same decoder; no claim of independent-object generalization.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    dest = FIGURES / 'recovery_depth_reduction.jpg'
    fig.savefig(dest, dpi=150, pil_kwargs={'quality': 94}); plt.close(fig)
    return dict(figure=str(dest), completed_cases=len(records), expected_cases=4,
        records=[dict(index=index, seed=seed, reductions={name: value['depth_error_reduction']
                    for name, value in pair.items()}) for index, seed, pair in records])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('recovery', 'factorial', 'all'), default='all')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--side', type=int, default=256)
    parser.add_argument('--indices', default=None)
    parser.add_argument('--seeds', default=','.join(map(str, PARAMS['seeds'])))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    setup(args.gpu)
    if args.side < 32: raise ValueError('figure side must be at least 32')
    import nvdiffrast.torch as dr
    device = torch.device('cuda:0')
    context = dr.RasterizeCudaContext(device=device)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest_path = FIGURES / 'manifest.json'
    manifest = read(manifest_path) if manifest_path.exists() else dict(recovery={}, factorial={})
    seeds = [int(seed) for seed in args.seeds.split(',')]
    skipped = []
    for phase in ('recovery', 'factorial'):
        if args.phase not in (phase, 'all'): continue
        indices = list(map(int, args.indices.split(','))) if args.indices else PARAMS['recovery_indices' if phase == 'recovery' else 'indices']
        for index in indices:
            for seed in seeds:
                key = f'{index:03d}_{seed}'
                previous = manifest[phase].get(key)
                if previous and Path(previous['figure']).exists() and not args.overwrite: continue
                record = (recovery_case if phase == 'recovery' else factorial_case)(index, seed, context, device, args.side)
                if record is None:
                    skipped.append(f'{phase}/{key}: missing completion receipt or endpoint'); continue
                manifest[phase][key] = record
                manifest.update(time=now(), status='partial', same_frame_and_cameras=True, model_loaded=False)
                write(manifest_path, manifest)
                gc.collect(); torch.cuda.empty_cache()
    if args.phase in ('recovery', 'all'): manifest['recovery_summary'] = recovery_summary()
    expected = dict(recovery=len(PARAMS['recovery_indices']) * len(PARAMS['seeds']),
                    factorial=len(PARAMS['indices']) * len(PARAMS['seeds']))
    count = {phase: len([record for record in manifest[phase].values() if Path(record['figure']).exists()])
             for phase in ('recovery', 'factorial')}
    complete = count == expected
    manifest.update(time=now(), status='complete' if complete else 'partial', expected_cases=expected,
                    rendered_cases=count, skipped_this_run=skipped, source=str(Path(__file__).resolve()),
                    same_frame_and_cameras=True, model_loaded=False)
    write(manifest_path, manifest)
    print(dict(stage='figures_finished', status=manifest['status'], rendered=count, expected=expected,
               skipped=skipped, manifest=str(manifest_path)), flush=True)


if __name__ == '__main__': main()
