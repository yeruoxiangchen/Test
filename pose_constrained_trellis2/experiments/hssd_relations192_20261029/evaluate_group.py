"""Full synchronous SS/Shape sampling; target meshes are read only after generation."""
import argparse
import gc
import math
import os
from . import common as c
from .train_group import seed_for


def run(arm, manifest, group_id, seed, smoke=False):
    os.environ.update(c.environment())
    import numpy as np
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from ...scene_relations.data import GroupDataset
    from ...scene_relations.model import attach_relations
    from ...scene_relations.sampling import sample_group
    from ...scene_relations.shape_support import ss_latent_to_shape_coords
    from ...conditioning import sample_to_device
    from ...models import load_trainable_state_dict
    from ... import runtime as rt
    dataset = GroupDataset(manifest, 'inference')
    index = next(i for i, g in enumerate(dataset.groups) if g['id'] == group_id)
    group = dataset.groups[index]
    samples = [sample_to_device(s, 'cuda') for s in dataset[index]]
    assert all(not any(k in s for k in ('target', 'target_coords', 'target_feats')) for s in samples)
    destination = c.OUT / ('smoke/evaluations' if smoke else 'evaluations') / arm / group_id / f'seed_{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    def release(): gc.collect(); torch.cuda.empty_cache()
    def model_for(stage):
        folder = c.OUT / ('smoke/training' if smoke else 'training') / arm / stage
        config = c.read(folder / 'complete.json')
        expected_manifest = c.DATA / ('smoke.json' if smoke else 'train.json')
        contract = config['contract']
        assert contract['arm'] == arm and contract['stage'] == stage and contract['smoke'] == smoke
        assert contract['runtime_identity'] == c.identity()['identity']
        assert contract['manifest_sha256'] == c.sha(expected_manifest)
        if not smoke:
            assert config['step'] == contract['train_groups'] * c.CONFIG['epochs']
        saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        assert saved['format'] == 't2_scene_group_v1' and saved['contract'] == contract
        model = rt.load_pose_conditioned_flow(stage, device='cuda', model_path=c.models()[stage + '_flow'],
                                             lora_rank=8, lora_alpha=16, gradient_checkpointing=False)
        if arm == 'relational': model = attach_relations(model)
        load_trainable_state_dict(model, saved['model_trainable_state'])
        checkpoints[stage] = config
        return model.eval()
    rt.SS_DECODER_MODEL = c.models()['ss_decoder']; rt.SHAPE_DECODER_MODEL = c.models()['shape_decoder']
    failures, meshes, shapes = {}, {}, {}
    with torch.no_grad():
        model = model_for('ss'); noises = []
        for sample in samples:
            torch.manual_seed(seed_for('ss', seed, sample['uid'])); torch.cuda.manual_seed_all(seed_for('ss', seed, sample['uid']))
            noises.append(torch.randn(1, 8, 16, 16, 16, device='cuda'))
        states = sample_group(model, 'ss', noises, samples, steps=12)
        del model, noises; release()
        decoder = rt.load_ss_decoder(device='cuda'); active, coords = [], []
        for state, sample in zip(states, samples):
            uid = sample['uid']; folder = destination / uid; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'ss.npz', z=state.float().cpu().numpy())
            try:
                coords.append(ss_latent_to_shape_coords(state, decoder, max_tokens=c.CONFIG['shape_max_tokens'])); active.append(sample)
            except RuntimeError as error:
                if 'empty support' not in str(error) and 'contract permits' not in str(error): raise
                failures[uid] = str(error)
        del decoder, states; release()
        if active:
            model = model_for('shape'); noises = []
            for sample, coord in zip(active, coords):
                value = seed_for('shape', seed, sample['uid']); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
                noises.append(rt.make_shape_noise(coord, device='cuda'))
            states = sample_group(model, 'shape', noises, active, steps=12)
            del model, noises; release()
            decoder = rt.load_shape_decoder(device='cuda'); decoder.low_vram = True
            for state, sample in zip(states, active):
                uid = sample['uid']; folder = destination / uid
                assert torch.isfinite(state.feats).all()
                np.savez_compressed(folder / 'shape.npz', coords=state.coords[:, 1:].cpu().numpy(),
                                    normalized_feats=state.feats.float().cpu().numpy())
                decoded = rt.decode_shape_latent(state, decoder)
                vertices = decoded.vertices.float().cpu().numpy(); faces = decoded.faces.cpu().numpy()
                if not len(vertices) or not len(faces): failures[uid] = 'empty decoded mesh'; continue
                assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
                mesh = trimesh.Trimesh(vertices, faces, process=False); mesh.export(folder / 'mesh.ply')
                world = mesh.copy(); world.apply_transform(sample['object_to_world'].cpu().numpy()); world.export(folder / 'mesh_world.ply')
                meshes[uid] = mesh
                del decoded, world
            del decoder, states; release()
    # Every object finishes sampling before any complete target mesh is read.
    results = []
    for sample in samples:
        uid = sample['uid']; metrics = {'cd_l1': math.sqrt(3.), 'f@0.01': 0., 'f@0.02': 0.}
        if uid in meshes:
            gt_path = sample['row'].get('gt_mesh', str(c.native.DATA / 'objects' / uid / 'target_t2.ply'))
            gt = trimesh.load(gt_path, force='mesh', process=False)
            np.random.seed(20261028); gt_points = gt.sample(100000)
            np.random.seed(20261029); pred_points = meshes[uid].sample(100000)
            a = cKDTree(pred_points).query(gt_points)[0]; b = cKDTree(gt_points).query(pred_points)[0]
            metrics['cd_l1'] = float((a.mean() + b.mean()) / 2)
            for threshold in (.01, .02):
                precision, recall = float((b < threshold).mean()), float((a < threshold).mean())
                metrics[f'f@{threshold}'] = 2 * precision * recall / max(precision + recall, 1e-12)
        T = sample['object_to_world'].cpu().numpy(); scale = np.linalg.svd(T[:3, :3], compute_uv=False)
        assert np.ptp(scale) < 1e-5
        metrics['world_cd_l1'] = metrics['cd_l1'] * float(scale.mean())
        results.append({'uid': uid, 'primary_anchor': uid == group['anchor'], 'success': uid in meshes,
                        'failure': failures.get(uid), 'metrics': metrics,
                        'mesh': str(destination / uid / 'mesh.ply') if uid in meshes else None,
                        'world_mesh': str(destination / uid / 'mesh_world.ply') if uid in meshes else None})
    c.write(destination / 'result.json', {'time': c.now(), 'arm': arm, 'group_id': group_id,
             'split': group['split'], 'scene': group['scene'], 'anchor': group['anchor'], 'seed': seed,
             'runtime_identity': c.identity()['identity'], 'manifest_sha256': c.sha(manifest),
             'objects': results, 'checkpoints': checkpoints, 'smoke': smoke,
             'sampling_steps': [12, 12], 'synchronous_updates': True,
             'gt_pose_controlled': True, 'fitted_alignment': False,
             'gt_mesh_read_after_group_generation': True,
             'failed_neighbors_removed_from_shape': sorted(failures),
             'failure_policy': 'preserve every anchor; no GT support substitution or token truncation'})
    print({'group': group_id, 'arm': arm, 'seed': seed, 'objects': len(results),
           'valid_meshes': sum(r['success'] for r in results)}, flush=True)
    if smoke and failures: raise RuntimeError('Smoke full sampling produced a structural failure; inspect before formal training')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=c.CONFIG['arms'], required=True)
    p.add_argument('--manifest', required=True); p.add_argument('--group', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.arm, a.manifest, a.group, a.seed, a.smoke)


if __name__ == '__main__': main()
