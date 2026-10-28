"""Post-protocol descriptive check: do both donor hypotheses share SS errors?"""
import numpy as np
from pathlib import Path

from visual_evidence_probe.common import OUT, folder, read, write, verify, now


def ids(coords):
    xyz = np.asarray(coords, np.int64)[:, -3:]
    return np.unique(xyz[:, 0] * 4096 + xyz[:, 1] * 64 + xyz[:, 2])


def main(output=None):
    out = Path(output) if output is not None else OUT
    indices = read(out / 'protocol.json')['indices']
    rows = []
    for index in indices:
        root = out / 'objects' / f'{index:03d}'
        rec = read(root / 'object.json')
        with np.load(verify(rec['gt_coords'])) as data:
            gt = ids(data['official_coords'])
        donors = []
        for seed in (42, 43):
            with np.load(root / 'results/real2' / f'{seed}_ss.npz') as data:
                donors.append(ids(data['coords']))
        a, b = donors
        errors = [np.setxor1d(value, gt) for value in donors]
        shared = np.intersect1d(*errors)
        union = np.union1d(*errors)
        fps = [np.setdiff1d(value, gt) for value in donors]
        fns = [np.setdiff1d(gt, value) for value in donors]
        rows.append({'index': index, 'uid': rec['uid'],
            'donor_support_iou': len(np.intersect1d(a, b)) / max(len(np.union1d(a, b)), 1),
            'error_set_jaccard': len(shared) / max(len(union), 1),
            'error_set_overlap_fraction_a': len(shared) / max(len(errors[0]), 1),
            'error_set_overlap_fraction_b': len(shared) / max(len(errors[1]), 1),
            'shared_false_positives': len(np.intersect1d(*fps)),
            'shared_false_negatives': len(np.intersect1d(*fns)),
            'error_count_a': len(errors[0]), 'error_count_b': len(errors[1])})
    result = {'time': now(), 'scope': 'post-protocol descriptive diagnostic, not an independence test',
              'metric': 'symmetric difference of donor SS support and original mesh surface support',
              'mean_error_set_jaccard': float(np.mean([r['error_set_jaccard'] for r in rows])), 'records': rows}
    write(out / 'source_error_overlap.json', result)
    print(result['mean_error_set_jaccard'])


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output')
    main(p.parse_args().output)
