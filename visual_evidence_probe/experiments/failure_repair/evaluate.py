import argparse
import gc

from .config import OUT, PARAMS, folder, data_folder, bind_engine, setup, register, read, write, now


def error_regions(original, target, region):
    from visual_evidence_probe.experiments.geometry_evidence.metrics import fixed_depth_error
    domain = original['mask'] | target['mask']
    bad = domain & (fixed_depth_error(original, target) >= PARAMS['bad_depth_threshold'])
    selected = domain & (region > .5)
    return dict(bad=bad, good=domain & ~bad, selected_bad=selected & bad, selected=selected, domain=domain)


def score_error_regions(pred, target, masks):
    from visual_evidence_probe.experiments.geometry_evidence.metrics import fixed_depth_error
    error = fixed_depth_error(pred, target)
    result = {}
    for name in ('bad', 'good', 'selected_bad'):
        mask = masks[name]
        result[name + '_depth_mae'] = float(error[mask].mean()) if bool(mask.any()) else None
        result[name + '_pixels'] = int(mask.sum())
    result['bad_fraction'] = int(masks['bad'].sum()) / max(int(masks['domain'].sum()), 1)
    result['selection_precision'] = int(masks['selected_bad'].sum()) / max(int(masks['selected'].sum()), 1)
    result['selection_recall'] = int(masks['selected_bad'].sum()) / max(int(masks['bad'].sum()), 1)
    return result


def one(index, context, device):
    import numpy as np
    import torch
    from visual_evidence_probe.experiments.dense_evidence.data import observations
    from visual_evidence_probe.experiments.dense_evidence.evaluation import load_target, render_depth
    from visual_evidence_probe.experiments.dense_evidence.gates import component_summary
    from visual_evidence_probe.experiments.geometry_evidence.targets import novel_cameras
    from visual_evidence_probe.experiments.geometry_evidence.metrics import Evaluator, face_region
    from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
    if not all((folder(index, s) / 'input_complete.json').exists() for s in PARAMS['seeds']):
        raise RuntimeError('both input-only seeds must be sealed before GT loading')
    rec, k, w, _, _ = observations(index, device)
    target = load_target(index)
    kn, wn, receipt = novel_cameras(k.cpu().numpy(), w.cpu().numpy(), 8)
    nk = torch.tensor(kn, device=device); nw = torch.tensor(wn, device=device)
    write(data_folder(index) / 'novel_cameras.json', dict(intrinsics=kn.tolist(), extrinsics=wn.tolist(), **receipt))
    gt = render_depth(target, k, w, context); ngt = render_depth(target, nk, nw, context)
    torch.save(dict(input={key:gt[key].cpu() for key in ('depth', 'mask', 'xyz')},
                    novel={key:ngt[key].cpu() for key in ('depth', 'mask', 'xyz')}), data_folder(index) / 'evaluation_targets.pt')
    for seed in PARAMS['seeds']:
        root = folder(index, seed)
        if (root / 'complete.json').exists():
            continue
        original = load_obj(root / 'original.obj')
        baseline = render_depth(original, k, w, context); novel = render_depth(original, nk, nw, context)
        with np.load(root / 'region_sealed.npz') as a:
            region = torch.tensor(a['region'], device=device); selected = torch.tensor(a['selected_faces'], device=device)
        novel_region = face_region(novel, selected)
        np.savez_compressed(root / 'novel_region.npz', region=novel_region.cpu().numpy())
        scorer = Evaluator(target, original, baseline, gt, region, novel, ngt, novel_region, k, w, nk, nw, context)
        fixed = error_regions(novel, ngt, novel_region)
        base = scorer(original, original=True, structure=component_summary(original.vertices, original.faces))
        base['fixed_error'] = score_error_regions(novel, ngt, fixed)
        write(root / 'baseline_metrics.json', base)
        cache = {(root / 'original.obj').resolve():base}
        scores = {}
        for endpoint, path in [('raw30k', root / 'native_original.obj')] + [
                (ep, root / 'full' / f'{ep}.obj') for ep in ('accepted', 'candidate', 'proposal')]:
            path = path.resolve()
            if path not in cache:
                mesh = load_obj(path)
                metric = scorer(mesh, structure=component_summary(mesh.vertices, mesh.faces))
                metric['fixed_error'] = score_error_regions(render_depth(mesh, nk, nw, context), ngt, fixed)
                cache[path] = metric
            scores[endpoint] = cache[path]
        source = read(root / 'dense' / 'complete.json'); selection = read(root / 'full' / 'selection.json')
        generation = read(root / 'baseline_complete.json')
        provider = read(root / 'dense_provider.json')
        depth_meta = read(__import__('pathlib').Path(provider['path']) / 'complete.json')
        value = dict(**selection, status=source['status'], original_metrics=base, metrics=scores,
                     input_score=read(root / 'input_score.json'),
                     timing=dict(generation_and_fit_seconds=generation['elapsed_seconds'],
                                 editor_seconds=source['elapsed_seconds'],
                                 depth_provider_seconds=depth_meta['elapsed_seconds'], depth_shared_across_seeds=True),
                     source_elapsed_seconds=source['elapsed_seconds'],
                     model_weights_frozen=True, gt_used_only_after_selection=True,
                     no_gt_alignment=True, fixed_error_masks_evaluation_only=True,
                     selected_checkpoint_diagnostic=source['decisions'])
        write(root / 'full' / 'result.json', value)
        write(root / 'complete.json', dict(time=now(), index=index, seed=seed, uid=rec['uid'],
              states=['raw30k', 'aligned30k', 'full'], gt_evaluation_only=True))
        print(dict(stage='evaluated_complete', index=index, seed=seed), flush=True)
        del scorer, cache, original, baseline, novel; gc.collect(); torch.cuda.empty_cache()
    del target, gt, ngt; gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='0'); p.add_argument('--indices', required=True)
    a = p.parse_args(); bind_engine(); setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device = runtime_setup(); context = dr.RasterizeCudaContext(device=device); register('evaluate')
    for index in map(int, a.indices.split(',')):
        one(index, context, device)


if __name__ == '__main__':
    main()
