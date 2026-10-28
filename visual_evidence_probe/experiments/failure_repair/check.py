"""Technical checks only: improvement never controls smoke success."""
import argparse
import math

from .config import OUT, PARAMS, folder, registry, read, write, now


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite evaluation value')
    if isinstance(value, dict):
        for v in value.values():finite_tree(v)
    if isinstance(value, list):
        for v in value:finite_tree(v)


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    reg=registry();uids={r['uid'] for r in reg['rows']}
    if uids & set(reg['excluded_uids']):raise ValueError('old cohort UID leaked')
    cases=[]
    for index in map(int,a.indices.split(',')):
        for seed in PARAMS['seeds']:
            root=folder(index,seed)
            if not read(root/'decoder_parity.json')['passed']:raise ValueError('decoder parity failed')
            if read(root/'input_complete.json')['gt_used']:raise ValueError('GT used by input stage')
            if read(root/'input_score.json')['gt_used']:raise ValueError('GT used by suspicion score')
            result=read(root/'full/result.json');finite_tree(result)
            if not result['input_only']:raise ValueError('non-input selection')
            original=torch.load(root/'original_latent.pt',map_location='cpu',weights_only=False)
            with np.load(root/'region_sealed.npz') as z:active=torch.tensor(z['active_rows'])
            verified=[]
            for endpoint in ('accepted','candidate','proposal'):
                current=torch.load(root/'full'/f'{endpoint}_latent.pt',map_location='cpu',weights_only=False)
                if not torch.equal(original['coords'],current['coords']):raise ValueError('SS coordinates changed')
                if not torch.equal(original['features'][~active],current['features'][~active]):raise ValueError('outside latent changed')
                if not bool(torch.isfinite(current['features']).all()):raise ValueError('nonfinite latent')
                if not (root/'full'/f'{endpoint}.obj').is_file():raise ValueError('missing mesh')
                verified.append(endpoint)
            cases.append(dict(index=index,seed=seed,verified=verified))
    receipt=dict(time=now(),passed=True,cases=cases,cohort_disjoint=True,
                 fixed_ss_and_outside_rows=True,finite_metrics=True,requires_metric_improvement=False)
    write(OUT/('smoke_complete.json' if a.smoke else 'validation/artifact_checks.json'),receipt)
    print(receipt,flush=True)


if __name__=='__main__':main()
