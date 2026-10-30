"""Small metadata freeze; expensive mask selection belongs to the user-run queue."""
import os
from collections import Counter
from . import common as c


def main():
    os.environ.update(c.environment())
    targets = c.read(c.SOURCE / 'targets.json')
    anchors = targets['plan']['objects']
    candidates = c.read(c.SOURCE / 'candidate_assets.json')
    families, geometry = {}, {}
    for r in candidates['objects']:
        families.setdefault(r['asset_family'], set()).add(r['split'])
        geometry.setdefault(r['candidate_geometry_hash'], set()).add(r['split'])
    eligible = [r for r in candidates['objects'] if len(families[r['asset_family']]) == 1
                and len(geometry[r['candidate_geometry_hash']]) == 1]
    assert dict(Counter(r['split'] for r in anchors)) == {'train': 128, 'val': 32, 'test': 32}
    assert {r['uid'] for r in anchors} <= {r['uid'] for r in eligible}
    reuse = {}
    for r in c.native.plan()['plan']['objects']:
        folder = c.native.DATA / 'objects' / r['uid']
        names = ['shape_512.npz', 'shape_1024.npz', 'ss.npz', 'transform.json',
                 'target_t2.ply', 'decoded_t2.ply', 'codec.json', 'encode_512.json']
        reuse[r['uid']] = {'folder': str(folder), 'source_mesh': r['mesh'],
                          'files': {n: (folder / n).stat().st_size for n in names}}
    pilot = c.read(c.ROOT / 'datasets/hssd_mv_baseline_v1/pair_proposals_v2.json')
    payload = {'source_targets_identity': targets['identity'],
               'source_targets_sha256': c.sha(c.SOURCE / 'targets.json'),
               'source_candidates_identity': candidates['identity'],
               'source_candidates_sha256': c.sha(c.SOURCE / 'candidate_assets.json'),
               'anchors': anchors, 'eligible_candidates': eligible,
               'pilot_groups': pilot['groups'], 'pilot_endpoints': pilot['endpoint_assets'],
               'native_reuse': reuse, 'config': c.CONFIG}
    c.immutable(c.OUT / 'plan.json', {'identity': c.digest(payload), 'plan': payload})
    # Independent short execution checks use two already encoded TRAIN objects.
    # They are not a replacement for later shared-frame pair data qualification.
    native = c.read(c.native.DATA / 'train.json')['samples']
    largest = max(native, key=lambda r: c.read(c.native.DATA / 'objects' / r['uid'] / 'encode_512.json')['tokens'])
    pair = [largest] + [r for r in native if r['scene'] == largest['scene'] and r['uid'] != largest['uid']][:1]
    assert len(pair) == 2 and all(r['split'] == 'train' for r in pair)
    group = {'id': 'native_train_interface', 'split': 'train', 'scene': pair[0]['scene'],
             'anchor': pair[0]['uid'], 'members': pair, 'scope': 'interface smoke only; individual cached views'}
    c.immutable(c.DATA / 'smoke.json', {'plan_identity': c.read(c.OUT / 'plan.json')['identity'], 'groups': [group]})
    print({'anchors': len(anchors), 'splits': dict(Counter(r['split'] for r in anchors)),
           'eligible_candidates': len(eligible), 'reusable_native': len(reuse),
           'missing_pilot_neighbors': pilot['missing_native_uids'], 'out': str(c.OUT)}, flush=True)


if __name__ == '__main__': main()
