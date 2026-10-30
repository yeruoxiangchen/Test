"""Bounded dense-support check on the actual frozen training pair; no new selection."""
import argparse
import math
import os
from pathlib import Path
from . import common as c

GROUP = 'hssd_103997403_171030405_i112'
UIDS = ('hssd_103997403_171030405_i086', GROUP)


def manifest():
    from .data_worker import group_for
    from ...scene_relations.data import GroupDataset
    group = group_for(GROUP)
    assert group['split'] == 'train' and set(group['uids']) == set(UIDS)
    members, diagnostics = [], []
    for uid in group['uids']:
        folder = c.codec_folder(uid); cond = c.DATA / 'conditions' / GROUP / uid
        codec = c.read(folder / 'codec.json'); ss = c.read(folder / 'ss.json')
        assert ss['iou'] >= .9 and codec['faces'] > 0
        assert codec['metrics']['cd_l1'] <= .04 and codec['metrics']['f@0.02'] >= .85, 'Inspect actual dense-support codec'
        files = {'pose_file': cond / 'pose.pt', 'ss_target_file': folder / 'ss.npz',
                 'shape_target_file': folder / 'shape_512.npz'}
        assert c.read(cond / 'condition.json')['frame_ids'] == group['frames']
        members.append({'uid': uid, 'split': 'train', 'scene': group['scene'], 'gt_mesh': str(folder / 'target_t2.ply'),
                        **{key: str(value) for key, value in files.items()},
                        **{key + '_sha256': c.sha(value) for key, value in files.items()}})
        diagnostics.append({'uid': uid, 'tokens': c.read(folder / 'encode_512.json')['tokens'],
                            'metrics': codec['metrics'], 'ss_iou': ss['iou']})
    assert max(r['tokens'] for r in diagnostics) > 8192
    document = {'plan_identity': c.read(c.OUT / 'plan.json')['identity'],
                'groups': [{**group, 'members': members}], 'scope': 'Dense-support execution check; not scientific improvement evidence'}
    c.immutable(c.DATA / 'support_probe.json', document)
    samples = GroupDataset(c.DATA / 'support_probe.json', 'shape', max_tokens=c.CONFIG['shape_max_tokens'])[0]
    assert [len(s['target_coords']) for s in samples] == [r['tokens'] for r in diagnostics]
    c.write(c.OUT / 'support_probe/codec_check.json', {'time': c.now(), 'objects': diagnostics,
             'shared_frames': group['frames'], 'complete_support': True})
    print(diagnostics, flush=True)


def verify():
    import json
    from ..hssd_mv_baseline_20261028 import common as previous
    assert previous.source_identity() == previous.read(previous.OUT / 'runtime.json'), 'Old baseline changed'
    first_losses, checks, peaks = [], [], []
    for arm in c.CONFIG['arms']:
        receipt = c.read(c.OUT / 'stages' / f'support_probe_{arm}.json')
        assert receipt['runtime_identity'] == c.identity()['identity']
        rows = [json.loads(line) for line in Path(receipt['log']).read_text().splitlines() if line.startswith('{')]
        assert [r['step'] for r in rows] == [1, 2]
        assert rows[0]['initial_stock_anchor'] == [1., 1.]
        assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in rows)
        if arm == 'relational': assert all(r['relation_gradient_norm'] > 0 for r in rows)
        first_losses.append(rows[0]['loss']); checks.append({'arm': arm, 'steps': rows})
        peaks.append(c.read(receipt['memory_report']))
    assert abs(first_losses[0] - first_losses[1]) < 1e-6
    for uid in UIDS:
        for name in ('encode_512', 'encode_1024', 'decode_512'):
            receipt = c.read(c.OUT / 'stages' / f'{uid}_{name}.json')
            peaks.append(c.read(receipt['memory_report']))
    assert all(r['status'] == 'complete' for r in peaks)
    c.write(c.OUT / 'support_probe/verification.json', {'time': c.now(), 'runtime_identity': c.identity()['identity'],
            'checks': checks, 'codec': c.read(c.OUT / 'support_probe/codec_check.json'),
            'first_step_losses_match': True, 'old_mvbase_source_unchanged': True,
            'max_allocated_mib': max(r['peak_allocated_mib'] for r in peaks),
            'max_reserved_mib': max(r['peak_reserved_mib'] for r in peaks),
            'scope': 'Complete native codec and two optimization steps on real dense training pair. Does not establish quality gains or memory capacity for every possible 32768-token object.'})
    print('Dense-support codec/gradient check passed; resume full frozen dataset.', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['manifest', 'verify']); a = p.parse_args()
    os.environ.update(c.environment())
    if a.action == 'manifest': manifest()
    else: verify()


if __name__ == '__main__': main()
