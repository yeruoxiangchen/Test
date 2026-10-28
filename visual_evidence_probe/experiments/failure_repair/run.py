import argparse
import gc
import time

from .config import OUT, PARAMS, folder, bind_engine, setup, register, write, read, now


def one(decoder, index, seed, context, device):
    import numpy as np
    import torch
    from visual_evidence_probe.experiments.dense_evidence.data import observations, load_latent, fit_transform
    from visual_evidence_probe.experiments.dense_evidence.evidence import build
    from visual_evidence_probe.experiments.dense_evidence.optimize import run_source, link, iou
    from visual_evidence_probe.experiments.editor_support.optimize import decode
    from visual_evidence_probe.experiments.editor_support.objective import freeze_target
    from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
    root = folder(index, seed)
    if (root / 'input_complete.json').exists():
        return
    start = time.monotonic()
    rec, k, w, images, masks = observations(index, device)
    z = load_latent(index, seed, device); fit = fit_transform(index, seed, device)
    with torch.no_grad():
        _, faces, world, pred = decode(decoder, z, fit, k, w, context)
        baseline = freeze_target(pred); original = load_obj(root / 'original.obj')
        same = tuple(original.vertices.shape) == tuple(world.shape) and tuple(original.faces.shape) == tuple(faces.shape)
        error = float(np.max(np.abs(original.vertices-world.cpu().numpy()))) if same else None
        passed = bool(same and np.array_equal(original.faces, faces.cpu().numpy()) and error < 4.e-7)
        write(root / 'decoder_parity.json', dict(passed=passed, max_vertex_error=error))
        if not passed:
            raise ValueError(f'decoder parity failed {index}/{seed}: {error}')
        record = build(index, seed, baseline, rec, masks)
        record.update(images=images, masks=masks)
        target = record['targets']['dense']; train = record['train']
        valid = target['depth_valid'] & (record['region'] > .5) & baseline['mask']
        valid[record['heldout']] = False
        discrepancy = (target['depth'] - baseline['depth']).abs()[valid]
        median = float(discrepancy.median()) if discrepancy.numel() else None
        mask_ious = iou(baseline['mask'], record['observed'], record['observation_valid'])
        silhouette = float(1 - np.mean(np.asarray(mask_ious)[train]))
        score = .5 * silhouette + .5 * min(median / PARAMS['input_depth_scale'], 1.) if median is not None else silhouette
        write(root / 'input_score.json', dict(time=now(), score=score, silhouette_disagreement=silhouette,
              local_median_depth_disagreement=median, supported_pixels=int(discrepancy.numel()),
              train_views=train, heldout_views=record['heldout'], gt_used=False,
              purpose='pre-edit ranking only; inherited final gate controls delivery'))
    del faces, world, pred
    result = run_source(decoder, z, fit, k, w, context, baseline, record, 'dense', original, index, seed)
    step = result['decisions']['observed_step']; path = root / 'full'; path.mkdir(exist_ok=True)
    for endpoint, mesh, latent in (
        ('accepted', f'step_{step:03d}.obj', f'step_{step:03d}_latent.pt'),
        ('candidate', 'train_selected.obj', 'train_selected_latent.pt'),
        ('proposal', 'proposal.obj', 'proposal_latent.pt')):
        link(path / f'{endpoint}.obj', root / 'dense' / mesh)
        link(path / f'{endpoint}_latent.pt', root / 'dense' / latent)
    write(path / 'selection.json', dict(index=index, seed=seed, uid=rec['uid'], split=rec['split'],
          kind=rec['kind'], source='dense', selected_step=step, accepted_edit=step > 0,
          candidate_step=result['decisions']['own_training_step'], input_only=True, uses_gt=False))
    write(root / 'input_complete.json', dict(time=now(), index=index, seed=seed, uid=rec['uid'],
          gt_used=False, arms=['full'], elapsed_seconds=time.monotonic()-start, original_ss_fixed=True))
    print(dict(stage='input_complete', index=index, seed=seed, accepted=step > 0), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='0'); p.add_argument('--indices', required=True)
    a = p.parse_args(); bind_engine(); setup(a.gpu)
    import torch
    import nvdiffrast.torch as dr
    from visual_evidence_probe.runtime import setup as runtime_setup
    from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
    from visual_evidence_probe.experiments.editor_support.data import load_decoder
    device = runtime_setup(); context = dr.RasterizeCudaContext(device=device)
    register('run'); decoder, binding = load_decoder(device); write(OUT / 'decoder_binding.json', binding)
    with FixedForwardNative() as policy:
        for index in map(int, a.indices.split(',')):
            for seed in PARAMS['seeds']:
                one(decoder, index, seed, context, device); gc.collect(); torch.cuda.empty_cache()
        write(OUT / 'editor_policy.json', policy.manifest())


if __name__ == '__main__':
    main()
