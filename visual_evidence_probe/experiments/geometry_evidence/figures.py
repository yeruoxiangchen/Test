import argparse
import gc

import numpy as np
import torch

from .config import ARMS, OUT, PARAMS, folder, setup, read, write, now
from .metrics import render_mesh, fixed_depth_error
from visual_evidence_probe.experiments.editor_support.objective import world_from_depth
from visual_evidence_probe.experiments.editor_support.figures import _plotting, _images, _grid


LABEL = dict(oracle_dense_normal='A: Dense GT depth + normals',
    oracle_dense_depth='A-depth: Dense GT depth only',
    oracle_sparse='B: GT depth at input query pixels', input_sparse='C: Input-estimated depth')


def image_inputs(index, seed, dest):
    from PIL import Image, ImageDraw
    from visual_evidence_probe.experiments.editor_support.data import observation
    rec, _, _, images, _ = observation(index, torch.device('cpu'))
    with np.load(folder(index, seed) / 'input_sealed.npz') as a:
        region, pixels = a['region'], a['pixel']
    panel, size = Image.new('RGB', (4 * 260, 300), 'white'), 256
    draw = ImageDraw.Draw(panel)
    for view, image in enumerate(images):
        array = np.array(image.resize((size, size)))
        color = np.array([245, 196, 50])
        chosen = region[view] > .5
        array[chosen] = (.73 * array[chosen] + .27 * color).astype(np.uint8)
        x = view * 260
        panel.paste(Image.fromarray(array), (x, 24))
        draw.text((x + 4, 5), f'Input {view}', fill='black')
        for v, y, px in pixels:
            if v == view: draw.ellipse((x + px - 1, y + 23, x + px + 1, y + 25), fill='red')
    draw.text((4, 285), f'{rec["uid"]} / seed {seed}: yellow=input-only local region; red=saved matcher pixels; GT not used', fill='black')
    dest.parent.mkdir(parents=True, exist_ok=True); panel.save(dest, quality=93)


@torch.no_grad()
def case(index, seed, context, device):
    import trimesh
    root = folder(index, seed)
    if not (root / 'complete.json').exists(): return None
    complete = read(root / 'complete.json')
    camera = read(folder(index) / 'novel_cameras.json')
    k = torch.tensor(camera['intrinsics'], device=device, dtype=torch.float32)
    w = torch.tensor(camera['extrinsics'], device=device, dtype=torch.float32)
    cache = torch.load(folder(index) / 'evaluation_targets.pt', map_location=device, weights_only=False)
    gt = cache['novel']; gt['xyz'] = world_from_depth(gt['depth'], k, w) * gt['mask'][..., None]
    with np.load(root / 'novel_region.npz') as a: region = a['region']
    views = np.argsort(-region.sum((1, 2)), kind='stable')[:3].tolist()
    results = {arm: read(root / arm / 'result.json') for arm in ARMS}
    from visual_evidence_probe.experiments.editor_support.data import observation
    from visual_evidence_probe.experiments.geometry_evidence.targets import resize_observed
    _, ik, iw, _, masks = observation(index, device)
    observed = resize_observed(masks, cache['input']) > .5
    silhouette_audit = {}
    rows = [dict(label='GT scan\nEvaluation only', pred={key: gt[key].cpu() for key in ('depth', 'mask', 'xyz')})]
    sources = [('Frozen30K\nOriginal input-mask global fit', root / 'original.obj')]
    for arm in ARMS:
        r = results[arm]
        before = r['original_metrics']['novel']['local_target_depth_mae']
        after = r['metrics']['selected']['novel']['local_target_depth_mae']
        change = f'{(before-after)/before:+.1%}' if before and after is not None else 'NA'
        label = f'{LABEL[arm]}\nselected step {r["best_step"]}; local depth {change}'
        if not r.get('actual_selected_edit'): label += '\nNO EDIT'
        sources.append((label, root / arm / 'selected.obj'))
    receipts = []
    for label, source in sources:
        mesh = trimesh.load(source, process=False, force='mesh', skip_materials=True)
        pred = render_mesh(mesh, k, w, context)
        input_pred = render_mesh(mesh, ik, iw, context)
        ious = ((input_pred['mask'] & observed).sum((1, 2)) /
            (input_pred['mask'] | observed).sum((1, 2)).clamp_min(1)).cpu().tolist()
        silhouette_audit[str(source)] = dict(observed_mask_iou_by_view=ious,
            observed_mask_iou_mean=float(np.mean(ious)))
        rows.append(dict(label=label, pred={key: pred[key].cpu() for key in ('depth', 'mask', 'xyz')}))
        receipts.append(dict(path=str(source), vertices=len(mesh.vertices), faces=len(mesh.faces),
            bounds=np.asarray(mesh.bounds).tolist()))
        del mesh, pred, input_pred
    crop_boxes = []
    for view in views:
        mask = (rows[0]['pred']['mask'][view] | rows[1]['pred']['mask'][view]).numpy()
        yy, xx = np.nonzero(mask)
        if len(yy):
            center_x, center_y = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
            width = min(256, int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1) * 1.15) + 4)
            x0, y0 = max(0, min(256-width, int(center_x-width/2))), max(0, min(256-width, int(center_y-width/2)))
            crop_boxes.append((x0, y0, x0+width, y0+width))
        else: crop_boxes.append((0, 0, 256, 256))
    for row in rows:
        row['images'] = [image[y0:y1, x0:x1] for image, (x0, y0, x1, y1) in zip(_images(row['pred'], views), crop_boxes)]
    dest = OUT / 'figures/meshes' / f'{index:03d}_{seed}.jpg'
    _grid(rows, views, f'{complete["uid"]}, seed {seed}',
        'Same NEW evaluation cameras and world frame | A/B are privileged diagnostics; C uses input images', dest,
        'Selected by each arm\'s training objective; these are candidates, not necessarily accepted deployment outputs.',
        view_labels=[f'New view {v}\nheld out from optimization' for v in views])
    plt = _plotting()
    fig, axes = plt.subplots(5, 3, figsize=(10.8, 11.2), squeeze=False)
    target = rows[0]['pred']; original = rows[1]['pred']
    domain = (region > .5) & target['mask'].numpy()
    base_error = fixed_depth_error(original, target).numpy()
    cap = max(float(np.quantile(base_error[domain], .95)), .005) if domain.any() else .02
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('white')
    for ri, row in enumerate(rows[1:]):
        error = fixed_depth_error(row['pred'], target).numpy()
        for ci, view in enumerate(views):
            axis = axes[ri, ci]
            image = axis.imshow(np.ma.masked_where(~target['mask'][view].numpy(), error[view]),
                cmap=cmap, vmin=0, vmax=cap, interpolation='nearest')
            if np.any(region[view]) and np.any(region[view] < .5):
                axis.contour(region[view], levels=[.5], colors=['#33ee88'], linewidths=.5)
            x0, y0, x1, y1 = crop_boxes[ci]
            axis.set_xlim(x0-.5, x1-.5); axis.set_ylim(y1-.5, y0-.5)
            axis.set_xticks([]); axis.set_yticks([])
            if ri == 0: axis.set_title(f'New view {view}')
            if ci == 0: axis.text(-.06, .5, row['label'], transform=axis.transAxes, ha='right', va='center', fontsize=7.8)
    fig.subplots_adjust(left=.30, right=.91, bottom=.045, top=.94, hspace=.03, wspace=.03)
    cax = fig.add_axes([.94, .25, .012, .48]); fig.colorbar(image, cax=cax, label='Absolute camera-z error')
    fig.suptitle(f'{complete["uid"]}, seed {seed}: errors on new evaluation views\nGreen outlines = fixed input-derived local region; same error scale for every row', fontsize=10)
    fig.text(.5, .012, f'Fixed GT target domain; missing rays pay0.1. Saturated above {cap:.4f}.', ha='center', fontsize=8)
    error_path = OUT / 'figures/errors' / dest.name
    error_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(error_path, dpi=125, pil_kwargs={'quality': 93}); plt.close(fig)
    input_path = OUT / 'figures/inputs' / dest.name
    image_inputs(index, seed, input_path)
    original_iou = silhouette_audit[str(root / 'original.obj')]
    for arm in ARMS:
        score = silhouette_audit[str(root / arm / 'selected.obj')]
        score['mean_iou_gain_to_observed'] = score['observed_mask_iou_mean'] - original_iou['observed_mask_iou_mean']
        score['accepted_by_legacy_gate'] = results[arm]['input_acceptance']['accepted']
    write(root / 'observed_silhouette_diagnostic.json', dict(posthoc_evaluation_only=True,
        changes_selection=False, legacy_gate_reference='original mesh', rows=silhouette_audit))
    del cache, gt, rows
    gc.collect(); torch.cuda.empty_cache()
    return dict(index=index, seed=seed, meshes=str(dest), errors=str(error_path), inputs=str(input_path),
        views=views, same_cameras_and_frame=True, privileged_oracle_rows=True,
        independent_objects=False, endpoints='selected', color_scale_max=cap, sources=receipts)


def summary():
    source = read(OUT / 'summary.json')
    plt = _plotting()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    colors = ['#7964a5', '#477ba8', '#42a181', '#c78536']
    items = [('input.local_depth_relative_reduction', 'Original input: local depth reduction (%)', 100),
        ('novel.local_depth_relative_reduction', 'New views: local depth reduction (%)', 100),
        ('surface.chamfer_l1', 'Global Chamfer improvement', 1)]
    for axis, (key, title, scale) in zip(axes, items):
        rows = [source['all_cases'][arm]['endpoints']['selected'][key] for arm in ARMS]
        value = np.array([r['mean'] or 0 for r in rows]) * scale
        low = np.array([(r['ci95'] or [0, 0])[0] for r in rows]) * scale
        high = np.array([(r['ci95'] or [0, 0])[1] for r in rows]) * scale
        axis.bar(np.arange(4), value, color=colors, yerr=np.maximum(np.stack((value-low, high-value)), 0), capsize=3)
        axis.axhline(0, color='#555555', linewidth=.7)
        axis.set_xticks(range(4), ['A\n+normal', 'A-depth', 'B\nGT sparse', 'C\nInput sparse'])
        axis.set_title(title, fontsize=9); axis.grid(axis='y', alpha=.2)
    fig.suptitle('Sixth probe: actual reconstruction errors, frozen30K (8 inspected objects x2 seeds)', fontsize=11)
    fig.text(.5, .015, 'Training-selected candidates; A/B are privileged diagnostics. Error bars: object bootstrap95% CI; no-ops retained.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .92))
    dest = OUT / 'figures/summary.jpg'; fig.savefig(dest, dpi=160, pil_kwargs={'quality': 94}); plt.close(fig)
    return str(dest)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='0'); p.add_argument('--indices', default=','.join(map(str, PARAMS['indices'])))
    p.add_argument('--seeds', default='42,43'); p.add_argument('--overwrite', action='store_true'); a = p.parse_args()
    setup(a.gpu)
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    dest = OUT / 'figures/manifest.json'
    manifest = read(dest) if dest.exists() else dict(cases={})
    for index in map(int, a.indices.split(',')):
        for seed in map(int, a.seeds.split(',')):
            key = f'{index:03d}_{seed}'
            if key in manifest['cases'] and not a.overwrite: continue
            record = case(index, seed, context, torch.device('cuda:0'))
            if record:
                manifest['cases'][key] = record
                write(dest, manifest)
                print(dict(stage='figure_complete', case=key), flush=True)
    if (OUT / 'summary.json').exists(): manifest['summary'] = summary()
    manifest.update(time=now(), complete=len(manifest['cases']) == 16, expected=16)
    write(dest, manifest)


if __name__ == '__main__': main()
