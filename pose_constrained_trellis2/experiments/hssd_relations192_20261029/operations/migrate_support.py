"""Explicit, locked migration of pre-training receipts; no model/data relabeling."""
import argparse
import fcntl
import shutil
from pathlib import Path
from .. import common as c

OLD = '10ac94b9021541c3faf330a5067548c79984d45df830c31b54c7e61e40deb3c0'
REV = c.OUT / 'revisions/shape_support_32768_20261029'
CHANGED = {
    'pose_constrained_trellis2/scene_relations/data.py',
    'pose_constrained_trellis2/scene_relations/shape_support.py',
    *('pose_constrained_trellis2/experiments/hssd_relations192_20261029/' + n for n in
      ('common.py', 'data_worker.py', 'evaluate_group.py', 'runner.py', 'run.sh',
       'PROTOCOL.md', 'tests.py', 'support_probe.py', 'train_group.py')),
}


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['snapshot', 'apply']); a = p.parse_args()
    with (c.OUT / 'launch.lock').open('a') as launch, (c.OUT / 'queue.lock').open('a') as queue:
        for lock in (launch, queue): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == 'snapshot':
            old = c.read(c.OUT / 'runtime.json'); assert old['identity'] == OLD
            assert not (c.OUT / 'training').exists(), 'Formal weights already exist; review separately'
            assert not REV.exists(), 'Snapshot already exists'
            REV.mkdir(parents=True)
            for name in old['runtime']['files']:
                src = c.REPO / name; dst = REV / 'source' / name
                assert c.sha(src) == old['runtime']['files'][name], name
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            for name in ('runtime.json', 'plan.json', 'status.json', 'gpu_policy.json'):
                shutil.copy2(c.OUT / name, REV / name)
            shutil.copytree(c.OUT / 'stages', REV / 'stages')
            print({'snapshot': str(REV), 'receipts': len(list((REV / 'stages').glob('*.json')))})
            return
        old = c.read(REV / 'runtime.json'); new = c.identity()
        assert old['identity'] == OLD and c.read(c.OUT / 'runtime.json') == old
        assert not (c.OUT / 'training').exists()
        assert old['runtime']['plan_identity'] == new['runtime']['plan_identity']
        assert old['runtime']['models'] == new['runtime']['models']
        expected_config = {**old['runtime']['config'], 'shape_max_tokens': 32768}
        assert new['runtime']['config'] == expected_config
        before, after = old['runtime']['files'], new['runtime']['files']
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        assert changed <= CHANGED, changed - CHANGED
        for name in changed:
            if name in before: assert c.sha(REV / 'source' / name) == before[name]
        from ..runner import preflight_jobs, data_jobs
        expected = {name: args for name, args, _, _ in list(preflight_jobs()) + list(data_jobs())}
        bindings = {}
        for file in sorted((c.OUT / 'stages').glob('*.json')):
            record = c.read(file); name = file.stem
            assert record == c.read(REV / 'stages' / file.name), name
            assert record['runtime_identity'] == OLD
            assert name in expected and record['command_identity'] == c.digest(expected[name]), name
            for row in record['artifacts']:
                path = Path(row['path']); assert path.is_file() and path.stat().st_size == row['bytes'], path
            bindings[name] = c.digest(record)
        record = {'time': c.now(), 'old_identity': OLD, 'new_identity': new['identity'],
                  'receipt_digests': bindings, 'changed_files': sorted(changed),
                  'archive': str(REV), 'training_weights_exist': False,
                  'reason': '8192 is a dataset filter, not native VAE support capacity. Keep complete 32^3 support.',
                  'compatibility': 'Same official codec math, labels and outputs below old limit; unchanged selection, seeds, models, losses. Old smoke remains attributed to old identity. New real dense-support check precedes continuation.'}
        c.write(c.OUT / 'receipt_compatibility.json', record)
        c.write(REV / 'migration.json', record)
        c.write(c.OUT / 'runtime.json', new)
        print({'new_identity': new['identity'], 'retained_receipts': len(bindings), 'changed_files': sorted(changed)})


if __name__ == '__main__': main()
