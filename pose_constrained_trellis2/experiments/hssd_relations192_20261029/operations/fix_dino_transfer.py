"""Archive and apply the exact DINO .to() interface repair with receipt lineage."""
import fcntl
import shutil
from pathlib import Path
from .. import common as c


def main():
    root = 'pose_constrained_trellis2/experiments/hssd_relations192_20261029/'
    replacements = {
        root + 'data_worker.py': (
            "    extractor = NativeDinoV3FeatureExtractor(c.models()['dino'], image_size=512).to('cuda')",
            "    extractor = NativeDinoV3FeatureExtractor(c.models()['dino'], image_size=512)\n    extractor.to('cuda')"),
        root + 'common.py': (
            "    assert migration['old_identity'] == receipt['runtime_identity'], 'Unknown prior source'",
            "    expected = migration.get('receipt_identities', {}).get(name, migration['old_identity'])\n    assert expected == receipt['runtime_identity'], 'Unknown prior source'"),
    }
    with (c.OUT / 'launch.lock').open('a') as launch, (c.OUT / 'queue.lock').open('a') as queue:
        for lock in (launch, queue): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = c.read(c.OUT / 'runtime.json')
        assert old['identity'] == '429ff02fcc1d87aa9b7da35dd739350c3161cb7f7b05a5508dc1f391841084c1'
        assert c.identity() == old and not (c.OUT / 'training').exists()
        prior = c.read(c.OUT / 'receipt_compatibility.json')
        bindings, identities = {}, {}
        for file in sorted((c.OUT / 'stages').glob('*.json')):
            record = c.read(file); c.check_receipt_identity(file.stem, record, old['identity'])
            for row in record['artifacts']:
                path = Path(row['path']); assert path.is_file() and path.stat().st_size == row['bytes']
            bindings[file.stem] = c.digest(record); identities[file.stem] = record['runtime_identity']
        content = {}
        for name, (before, after) in replacements.items():
            text = (c.REPO / name).read_text(); assert text.count(before) == 1
            content[name] = text.replace(before, after)
        archive = c.OUT / 'revisions/dino_transfer_20261029'; assert not archive.exists()
        archive.mkdir(parents=True)
        for name in replacements:
            dest = archive / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.REPO / name, dest)
        for name in ('runtime.json', 'receipt_compatibility.json', 'status.json'):
            shutil.copy2(c.OUT / name, archive / name)
        shutil.copytree(c.OUT / 'stages', archive / 'stages')
        for name, text in content.items(): (c.REPO / name).write_text(text)
        new = c.identity()
        changed = {k for k, v in new['runtime']['files'].items() if old['runtime']['files'].get(k) != v}
        assert changed == set(replacements)
        assert old['runtime']['config'] == new['runtime']['config']
        record = {'time': c.now(), 'old_identity': prior['old_identity'], 'new_identity': new['identity'],
                  'receipt_identities': identities, 'receipt_digests': bindings, 'archive': str(archive),
                  'previous_migration': str(archive / 'receipt_compatibility.json'),
                  'reason': 'DinoV3FeatureExtractor.to returns None. Keep extractor and transfer separately. Completed codec and prior smoke math unchanged.'}
        c.write(c.OUT / 'receipt_compatibility.json', record); c.write(archive / 'migration.json', record)
        c.write(c.OUT / 'runtime.json', new)
        print({'new_identity': new['identity'], 'retained_receipts': len(bindings)})


if __name__ == '__main__': main()
