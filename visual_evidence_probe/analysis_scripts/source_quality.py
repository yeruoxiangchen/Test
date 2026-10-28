"""Post-hoc input-only baseline: select donor using its two observed masks."""
import numpy as np
from PIL import Image
from pathlib import Path

from visual_evidence_probe.common import OUT, folder, read, write, verify, now
from visual_evidence_probe.render import mask_iou


def main(output=None):
    out = Path(output) if output is not None else OUT
    protocol = read(out / 'protocol.json')
    indices = protocol['indices']
    render_dir = 'renders_raw' if protocol.get('cropping') else 'renders'
    rows = []
    for index in indices:
        root = out / 'objects' / f'{index:03d}'
        record = read(root / 'object.json')
        observed = np.stack([np.asarray(Image.open(verify(b)).convert('L')) > 127 for b in record['masks'][:2]])
        values = {}
        for key in ('a', 'b'):
            with np.load(root / render_dir / key / 'masks.npz') as data:
                predicted = data['masks'][:2]
            values[key] = mask_iou(predicted, observed).tolist()
        best = max(('a', 'b'), key=lambda key: float(np.mean(values[key])))
        row = {'index': index, 'uid': record['uid'], 'observed_mask_iou': values,
               'selected_donor': best, 'selection_inputs': 'two original masks + known cameras; no GT mesh'}
        paths = {arm: [root / 'results' / arm / f'{seed}.json' for seed in (42, 43)]
                 for arm in ('donor_a6', 'donor_b6')}
        if all(p.exists() for paths_a in paths.values() for p in paths_a):
            cd = {arm: float(np.mean([read(p)['surface']['chamfer_l1'] for p in paths_a])) for arm, paths_a in paths.items()}
            row.update(cd=cd, selected_cd=cd['donor_a6' if best == 'a' else 'donor_b6'],
                       random_donor_mean_cd=float(np.mean(list(cd.values()))))
        rows.append(row)
    write(out / 'source_input_consistency.json', {'time': now(),
        'scope': 'post-hoc diagnostic baseline, threshold-free, no fitted policy',
        'selection_uses_gt_mesh': False, 'records': rows})
    print([(r['index'], r['selected_donor'], {k: float(np.mean(v)) for k, v in r['observed_mask_iou'].items()}) for r in rows])


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output')
    main(p.parse_args().output)
