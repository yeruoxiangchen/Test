"""CPU-only point/cell geometry audit using previously hash-bound LaRI caches."""
import argparse
import json
import time
import torch
from ...v2.dataset import SceneDataset
from ..hssd_full_diagnostic_20261021.diagnose import load_prediction, oracle_prediction, query_sets
from .common import OUT, PILOT, PACKAGE, binding, verify, sources, write
from .query import geometry_features


def audit(end=16):
    torch.set_num_threads(2)
    old = PACKAGE / 'outputs/hssd_full_diagnostic_20261021'
    source_identity = sources()
    records = []
    for domain in ['clean', 'occluded']:
        dataset = SceneDataset(PILOT / f'{domain}16.json', require_labels=True)
        for index in range(min(end, len(dataset))):
            begin = time.monotonic()
            prior_path = old / 'geometry' / domain / f'{index:03d}.json'
            prior = json.loads(prior_path.read_text())
            verify(prior['prediction'])
            verify(prior['identity']['manifest'])
            verify(prior['identity']['checkpoint'])
            ident = {'sources': source_identity, 'prediction': prior['prediction'],
                     'prior_record': binding(prior_path), 'manifest': binding(dataset.path)}
            path = OUT / 'geometry' / domain / f'{index:03d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                if row['identity'] != ident:
                    raise ValueError('Changed geometry diagnostic identity')
                records.append(row)
                continue
            sample = dataset[index]
            pred = load_prediction(verify(prior['prediction']), 'cpu')
            gt = oracle_prediction(sample, 'cpu')
            p16, _, shell, far = query_sets(sample, 'cpu')
            groups = {'target_surface_cells': shell, 'far_from_target_cells': far}
            stats = {}
            for name, prediction in [('oracle', gt), ('predicted', pred)]:
                for kind in ['point', 'cell_mean_27']:
                    f = geometry_features(sample, prediction, p16, kind=kind)
                    best = f[..., 0].amax(0)
                    stats[name + '_' + kind] = {key: {'count': int(mask.sum()),
                        'mean_max_view_support': float(best[mask].mean()) if mask.any() else None,
                        'fraction_support_gt_01': float((best[mask] > .1).float().mean()) if mask.any() else None}
                        for key, mask in groups.items()}
            row = {'uid': sample['uid'], 'index': index, 'domain': domain, 'identity': ident,
                'stats': stats, 'elapsed_seconds': time.monotonic() - begin,
                'scope': 'support proxy on GT64-derived coarse cells; not SS prediction, mesh, or final quality gate'}
            write(path, row)
            records.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'identity'}), flush=True)
    if end == 16:
        if len(records) != 32:
            raise ValueError('Incomplete cohort')
        means = {}
        for domain in ['clean', 'occluded']:
            rr = [r for r in records if r['domain'] == domain]
            means[domain] = {variant: {group: {key: sum(r['stats'][variant][group][key] for r in rr)/len(rr)
                for key in ['mean_max_view_support', 'fraction_support_gt_01']}
                for group in ['target_surface_cells', 'far_from_target_cells']}
                for variant in rr[0]['stats']}
        write(OUT / 'geometry_summary.json', {'complete': True, 'objects_per_domain': 16,
              'means': means, 'scope': 'diagnostic only; no threshold-based automatic expansion'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--end', type=int, default=16)
    a = p.parse_args()
    if not 1 <= a.end <= 16: p.error('--end must be 1..16')
    audit(a.end)
