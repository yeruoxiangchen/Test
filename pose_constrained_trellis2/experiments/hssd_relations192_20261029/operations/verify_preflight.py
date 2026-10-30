"""Read-only preflight checks, then a small /data verification receipt."""
from pathlib import Path
from .. import common as c


def main():
    import json
    import math
    import numpy as np
    runtime = c.identity()['identity']
    assert c.read(c.OUT / 'runtime.json')['identity'] == runtime
    from ...hssd_mv_baseline_20261028 import common as previous
    assert previous.source_identity() == previous.read(previous.OUT / 'runtime.json'), 'Old baseline source changed'
    first_losses = {}; checks = []; peaks = []
    for arm in c.CONFIG['arms']:
        first_losses[arm] = {}
        for stage in ('ss', 'shape'):
            receipt = c.read(c.OUT / 'stages' / f'smoke_{arm}_{stage}.json')
            c.check_receipt_identity(f'smoke_{arm}_{stage}', receipt, runtime)
            lines = [json.loads(line) for line in Path(receipt['log']).read_text().splitlines() if line.startswith('{')]
            assert [r['step'] for r in lines] == [1, 2]
            assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in lines)
            assert lines[0]['initial_stock_anchor'] == [1., 1.]
            if arm == 'relational': assert all(r['relation_gradient_norm'] > 0 for r in lines)
            first_losses[arm][stage] = lines[0]['loss']
            peaks.append(c.read(receipt['memory_report']))
            checks.append({'arm': arm, 'stage': stage, 'steps': 2,
                           'first_loss': lines[0]['loss'], 'second_loss': lines[-1]['loss'],
                           'relation_gradient_norm': lines[-1]['relation_gradient_norm']})
        receipt = c.read(c.OUT / 'stages' / f'smoke_sample_{arm}.json')
        result = c.read(c.OUT / 'smoke/evaluations' / arm / 'native_train_interface/seed_42/result.json')
        assert result['sampling_steps'] == [12, 12] and result['synchronous_updates']
        assert len(result['objects']) == 2 and all(r['success'] for r in result['objects'])
        assert result['gt_mesh_read_after_group_generation'] and not result['fitted_alignment']
        for row in result['objects']:
            folder = Path(row['mesh']).parent
            for name in ('mesh.ply', 'mesh_world.ply', 'shape.npz', 'ss.npz'): assert (folder / name).stat().st_size > 0
            with np.load(folder / 'shape.npz') as payload:
                assert 0 < len(payload['coords']) <= 8192 and payload['normalized_feats'].shape[1] == 32
                assert np.isfinite(payload['normalized_feats']).all()
        peaks.append(c.read(receipt['memory_report']))
    for stage in ('ss', 'shape'):
        assert abs(first_losses['independent'][stage] - first_losses['relational'][stage]) < 1e-6
    groups = c.read(c.DATA / 'scene_groups/104348133_171513054.json')['groups']
    assert len(groups) == 8 and all(g['split'] == 'train' and len(g['uids']) == 2 for g in groups)
    record = {'time': c.now(), 'runtime_identity': runtime, 'checks': checks,
              'first_step_losses_match': True, 'complete_sampling_meshes': 4, 'old_mvbase_source_unchanged': True,
              'max_allocated_mib': max(r['peak_allocated_mib'] for r in peaks),
              'max_reserved_mib': max(r['peak_reserved_mib'] for r in peaks),
              'scene_mask_probe': {'anchors': 8, 'paired_groups': 8},
              'scope': 'Execution check only; formal 192-anchor data/training/evaluation not started'}
    c.write(c.OUT / 'smoke/verification.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
