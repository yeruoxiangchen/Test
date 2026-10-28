"""Audit support/condition separation and synchronize cameras without generating data."""
import numpy as np
import torch

from .common import *
from ..hssd_input_adaptation_20261025.evaluate import load_input
from ...evaluate_ss_occupancy import occupancy_metrics


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return a == b


def partition_counts(row):
    return {key: val['gt_samples'] for key, val in row['partitioned'].items()
            if isinstance(val, dict) and 'gt_samples' in val}


def main():
    torch.set_num_threads(2)
    ident = freeze_runtime()
    cache, input_audits, coordinate_audits = {}, [], []
    p = plan()['plan']
    for obj in p['objects']:
        source = {}; selected = {}; bounds = {}
        for level in LEVELS:
            source[level], selected[level], bounds[level] = load_input(obj, variant(level), cache)
        for key in ['target', 'target_coords', 'intrinsics', 'extrinsics', 'image_size',
                    'extrinsics_type', 'camera_forward_sign', 'grid_transform', 'canonical_from_world']:
            if key in source['518'] or key in source['2072']:
                if key not in source['518'] or key not in source['2072'] or not equal(source['518'][key], source['2072'][key]):
                    raise ValueError('Cross-domain geometry/GT mismatch: '+key)
        expected_frames = [base.FRAMES[i] for i in obj['visible_indices']]
        for level in LEVELS:
            if bounds[level]['selected_frames'] != expected_frames:
                raise ValueError('Changed observation frame selection')
            if selected[level]['stock_condition'].shape[0] != 1:
                raise ValueError('Unexpected SS context batch')
            if len(selected[level]['slat_condition']['cond']) != len(expected_frames):
                raise ValueError('SLat view/context desynchronization')
        if torch.equal(selected['518']['masked_visual_patch_features'], selected['2072']['masked_visual_patch_features']):
            raise ValueError('Expected distinct low/high SLat observations')
        input_audits.append({'uid': obj['uid'], 'original_index': obj['original_index'],
            'inputs': bounds, 'same_geometry_and_gt': True, 'condition_features_differ': True,
            'selected_frames': expected_frames})
        for seed in SEEDS:
            supports = {}
            count_ref = None
            for level in LEVELS:
                case = next(c for c in p['cases'] if case_key(c) == (obj['uid'], seed, level, level))
                mp, mesh, sp, ss = get_result(case)
                if ss['inputs'] != bounds[level] or mesh['inputs'] != bounds[level]:
                    raise ValueError('Archived input binding differs from actual observations')
                for field in ['mesh', 'world_mesh', 'gt', 'ss']:
                    verify(mesh[field])
                with np.load(verify(ss['coords'])) as data:
                    coords = data['coords']
                if (coords.dtype != np.int32 or coords.ndim != 2 or coords.shape[1] != 4
                        or not len(coords) or np.any(coords[:,0] != 0)
                        or np.any(coords[:,1:] < 0) or np.any(coords[:,1:] >= 64)
                        or not np.array_equal(np.unique(coords, axis=0), coords)):
                    raise ValueError('Invalid or unsorted canonical support')
                if occupancy_metrics(coords, source[level]['target_coords'].numpy()) != ss['occupancy']:
                    raise ValueError('Archived SS metrics differ')
                counts = partition_counts(mesh)
                expected = dict(zip(['visible', 'externally_occluded', 'other_unobserved'],
                    [5247, 9477, 5276] if obj['uid'].endswith('i075') else [5691, 14309, 0]))
                if counts != expected:
                    raise ValueError('Unexpected archived GT partition counts')
                if count_ref is not None and counts != count_ref:
                    raise ValueError('Changed evaluation partitions')
                count_ref = counts
                supports[level] = coords
            codes = {level: np.ravel_multi_index(supports[level][:,1:].T, (64,64,64)) for level in LEVELS}
            coordinate_audits.append({'uid':obj['uid'], 'seed':seed,
                'counts':{level:len(supports[level]) for level in LEVELS},
                'common_coordinates':len(np.intersect1d(codes['518'], codes['2072'])),
                'partition_counts':count_ref})
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation unexpectedly initialized CUDA')
    write(OUT/'preflight.json', {'time':now(), 'identity':ident, 'passed':True,
        'input_audits':input_audits, 'coordinate_audits':coordinate_audits,
        'new_jobs':12, 'reference_jobs':12, 'cuda_initialized':False,
        'same_coordinate_noise_contract':'GPU smoke verifies full original-index master field and support intersections'})
    event('prepare_complete', new_meshes=12, referenced_meshes=12, input_caches=4)


if __name__ == '__main__':
    main()
