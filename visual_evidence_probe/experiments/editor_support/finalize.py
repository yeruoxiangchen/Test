"""Verify all saved endpoints, enrich small metadata, then append the fifth audit."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .config import OUT, STORE, PRIOR, PARAMS, read, write, now, digest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--wait', action='store_true'); a = parser.parse_args()
    expected = [OUT / 'factorial' / f'{i:03d}' / str(s) / 'complete.json'
        for i in PARAMS['indices'] for s in PARAMS['seeds']]
    while not all(p.exists() for p in expected):
        if not a.wait: raise RuntimeError('factorial cases incomplete')
        alive = False
        for gpu in (0, 2, 3):
            status = subprocess.run(['systemctl', '--user', 'is-active',
                f'visual-evidence-editor-factorial-g{gpu}-v2-20261026.service'], text=True, capture_output=True)
            alive |= status.stdout.strip() == 'active'
        if not alive: raise RuntimeError('all factorial workers stopped before completing16 cases')
        time.sleep(10)
    from visual_evidence_probe.analysis_scripts.local_repair_latent_report import support_fraction
    state_count = 0
    exact_fallback_count = 0
    coverages = []
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            case = OUT / 'factorial' / f'{index:03d}' / str(seed)
            original = torch.load(PRIOR / 'latents' / f'{index:03d}_{seed}.pt', map_location='cpu', weights_only=False)
            support_path = case / 'expansion/support.json'
            if support_path.exists():
                support = read(support_path)
                with np.load(case / 'input_pool.npz') as pool:
                    targets = pool['native_targets']
                added = np.asarray(support['added_coordinates'], np.int32).reshape(-1, 3)
                before = original['coords'].numpy()
                after = np.concatenate((before, np.c_[np.zeros(len(added), np.int32), added]))
                coverage = dict(index=index, seed=seed,
                    original=support_fraction(before, targets), expanded=support_fraction(after, targets),
                    gt_used=False, posthoc_diagnostic_only=True)
                write(case / 'expansion/target_corner_coverage.json', coverage); coverages.append(coverage)
                if support['added_tokens'] == 0 and not (case / 'expansion/initialization.json').exists():
                    write(case / 'expansion/initialization.json', dict(status='no_support_change',
                        preservation_steps=0, original_features_exact=support['original_features_exact'],
                        original_coordinates_exact=support['original_coordinates_exact'],
                        note='No new coordinates; per-arm step0 geometry metrics are saved. No unrecorded drift values are imputed.'))
            for support in ('original', 'expanded'):
                for objective in ('proxy', 'mesh'):
                    arm = case / f'{support}_{objective}'
                    record = read(arm / 'result.json')
                    if record['status'] == 'failed': raise RuntimeError(f'failed arm: {arm}')
                    if not record['accepted_edit']:
                        # This is the intended zero-edit fallback, not a fresh decode.
                        latent_path = arm / 'accepted_latent.pt'
                        cache = torch.load(latent_path, map_location='cpu', weights_only=False)
                        if not torch.equal(cache['coords'], original['coords']) or not torch.equal(cache['features'], original['features']):
                            raise ValueError(f'zero-edit latent differs from original: {arm}')
                        record['accepted_structure'] = record['original_structure']
                        record['metrics']['accepted'] = record['original_metrics']
                        record['exact_original_fallback'] = True
                        exact_fallback_count += 1
                    else:
                        record['accepted_structure'] = record['candidate_selected_structure']
                    for name in ('accepted.obj', 'proposal.ply', 'step0.ply', 'accepted_latent.pt'):
                        if not (arm / name).exists(): raise FileNotFoundError(arm / name)
                    write(arm / 'result.json', record); state_count += 1
    hardware = [dict(path=str(p), metadata=read(p)) for p in sorted(OUT.glob('hardware_*.json'))]
    write(OUT / 'hardware.json', dict(records=hardware, provenance='worker saved CUDA metadata'))
    failure_units = ['visual-evidence-editor-support-preflight-20261026',
        'visual-evidence-editor-recovery-asparagus-20261026', 'visual-evidence-editor-recovery-carrot-20261026']
    failure_units += [f'visual-evidence-editor-factorial-g{g}-20261026' for g in (0, 2, 3)]
    for unit in failure_units:
        result = subprocess.run(['journalctl', '--user', '-u', unit + '.service', '--no-pager'], text=True, capture_output=True, check=True)
        p = OUT / 'failure_logs' / f'{unit}.txt'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(result.stdout)
    from .report import generate
    summary = generate(final=True)
    if not summary['complete']: raise RuntimeError(f'report not complete: {summary["schema_errors"]}')
    write(OUT / 'completion.json', dict(time=now(), recovery_states=8, factorial_states=state_count,
        exact_original_fallbacks=exact_fallback_count, all_results_complete=True,
        forward_cuda_parity='historical FixedNative == FixedForwardNative, bitwise',
        latent_backward='finite nonzero with recorded static loss scaling',
        target_corner_coverage=coverages, failures_preserved=True,
        audit_appended=True, runtime_data_root=str(STORE),
        source=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__))))
    print(dict(stage='fifth_round_numerical_complete', recovery=8, factorial=state_count,
        report=str(OUT / 'REPORT.md')), flush=True)


if __name__ == '__main__': main()
