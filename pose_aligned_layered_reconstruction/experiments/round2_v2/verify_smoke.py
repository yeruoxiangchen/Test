"""Read generated artifacts and persist a reproducible implementation QA report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from ...paired_dataset import sha256_file
from ...v2.dataset import SceneDataset
from ...v2.checkpoints import read_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='pose_aligned_layered_reconstruction/outputs/round2_v2_verified_20261017')
    a = p.parse_args()
    root = Path(a.root).resolve()
    report = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'implementation_smoke_not_accuracy_experiment'}
    runs = {}
    for stage, folder, steps in [('lari', 'lari_current', 2), ('ss', 'ss', 2), ('slat', 'slat_final', 2),
                                 ('new_data_ss', 'ss_new_data', 3), ('ddp', 'lari_ddp_verified', 1)]:
        path = root / folder
        complete = json.loads((path / 'complete.json').read_text())
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
        checkpoint = read_checkpoint(path / 'checkpoints/last.pt')
        if complete['step'] != steps or checkpoint['step'] != steps:
            raise ValueError(f'incomplete training: {folder}')
        if not all(np.isfinite(row['loss']) and np.isfinite(row['gradient_norm']) for row in history):
            raise ValueError('nonfinite training')
        runs[stage] = {'updates': steps, 'last_metrics': history[-1],
            'checkpoint_sha256': sha256_file(path / 'checkpoints/last.pt')}
        if stage == 'ddp':
            hashes = complete['final_rank_parameter_hashes']
            if len(hashes) != 2 or len(set(hashes)) != 1:
                raise ValueError('DDP weights are not identical')
            for rank in range(2):
                if not (path / f'checkpoints/last.pt.rank{rank}.rng.pt').is_file():
                    raise ValueError('DDP RNG checkpoint missing')
            runs[stage]['identical_rank_parameters'] = True
    report['training'] = runs
    original = read_checkpoint(root / 'lari_current/checkpoints/last.pt')
    resumed = read_checkpoint(root / 'lari_resume/checkpoints/last.pt')
    exact = all(torch.equal(value, resumed['model_state'][key]) for key, value in original['model_state'].items())
    ema_exact = all(torch.equal(value, resumed['ema_state'][key]) for key, value in original['ema_state'].items())
    maximum = max(float((value - resumed['model_state'][key]).abs().max())
                  for key, value in original['model_state'].items())
    # CUDA grid_sample backward uses atomic accumulation. Save/restore is exact,
    # but repeating that backward kernel can differ by FP32 rounding units.
    if maximum > 1.e-8 or not ema_exact:
        raise ValueError('resume exceeds tight FP32 implementation tolerance')
    report['resume'] = {'model_bit_exact': exact, 'model_max_abs': maximum,
                        'model_atol': 1.e-8, 'ema_bit_exact': ema_exact,
                        'note': 'CUDA grid_sample backward is not bit-deterministic'}
    report['datasets'] = {}
    for name in ('paired_dataset', 'legacy_occlusion'):
        dataset = SceneDataset(root / name / 'manifest.json', require_ss_target=True)
        samples = [dataset[i] for i in range(len(dataset))]
        report['datasets'][name] = {'samples': len(dataset), 'objects': len(dataset.identity()['object_uids']),
            'domains': sorted({s['data_domain'] for s in samples}),
            'non_target_layers': sum(int(((s['layer_target_gt'] == 0) & (s['layer_depths_gt'] > 0)).sum()) for s in samples)}
        if name == 'paired_dataset':
            if len(samples) != 2:
                raise ValueError('expected clean/occluded pair')
            first, second = samples
            if not (torch.equal(first['intrinsics'], second['intrinsics'])
                    and torch.equal(first['extrinsics'], second['extrinsics'])
                    and torch.equal(first['target'], second['target'])):
                raise ValueError('paired clean/occluded geometry or camera differs')
            report['datasets'][name]['same_camera_and_target'] = True
        if name == 'legacy_occlusion':
            report['datasets'][name]['valid_supervision_fraction'] = float(samples[0]['supervision_valid_mask'].float().mean())
    scene = json.loads((root / 'full_scene_current/ss_target.npz.json').read_text())
    report['new_asset_ss_roundtrip_iou'] = scene['roundtrip_iou']
    mesh_complete = json.loads((root / 'mesh/complete.json').read_text())
    if not mesh_complete['records'] or mesh_complete['records'][0]['faces'] < 1:
        raise ValueError('mesh inference failed')
    mesh_path = root / 'mesh/00000/mesh_canonical.obj'
    if not mesh_path.is_file():
        raise ValueError('OBJ missing')
    report['mesh'] = {**mesh_complete['records'][0], 'obj': str(mesh_path), 'obj_sha256': sha256_file(mesh_path)}
    isolated = json.loads((root / 'mesh_isolated/complete.json').read_text())
    isolated_obj = root / 'mesh_isolated/00000/mesh_canonical.obj'
    isolated_ss = json.loads((root / 'ss_isolated/complete.json').read_text())
    if isolated_ss['step'] != 1 or not isolated['records'][0]['faces']:
        raise ValueError('isolated V2 training/inference incomplete')
    report['isolated_v2'] = {'ss_updates': isolated_ss['step'], 'mesh': isolated['records'][0],
        'obj': str(isolated_obj), 'obj_sha256': sha256_file(isolated_obj),
        'obj_same_as_before_isolation': sha256_file(isolated_obj) == sha256_file(mesh_path)}
    package = Path(__file__).resolve().parents[2]
    frozen = json.loads((package / 'outputs/mesh_comparison_20260910/dev64_v2/plan.json').read_text())['plan']['code_sha256']
    report['v1_source_hashes_unchanged'] = {name: sha256_file(package / name) == frozen[name]
        for name in ('native_ss.py', 'frozen_slat.py', 'contracts.py')}
    if not all(report['v1_source_hashes_unchanged'].values()):
        raise ValueError('frozen v1 core source changed')
    report['passed'] = True
    (root / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
