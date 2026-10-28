"""CPU audit: official GT support, coordinates, conditions and prior references."""
import numpy as np
import torch
from .common import *
from ..hssd_input_adaptation_20261025.evaluate import load_input
from ...evaluate_ss_occupancy import occupancy_metrics


def partition_counts(row):
    return {key:val['gt_samples'] for key,val in row['partitioned'].items()
            if isinstance(val,dict) and 'gt_samples' in val}


def main():
    torch.set_num_threads(2)
    ident = freeze_runtime()
    p, cache, inputs_checked, pairs = plan()['plan'], {}, [], []
    for obj in p['objects']:
        source,sample,inputs = load_input(obj,VARIANT,cache)
        with np.load(verify(obj['ss'])) as data:
            target_coords = canonical_support(data['official_coords'])
            if not np.array_equal(data['target'],source['target'].numpy()):
                raise ValueError('SS target encoding differs from source')
        with np.load(verify(obj['slat'])) as data:
            if not np.array_equal(np.unique(data['coords'],axis=0),target_coords[:,1:]):
                raise ValueError('GT SLat support and official SS coordinates disagree')
        if not np.array_equal(canonical_support(source['target_coords'].numpy()),target_coords):
            raise ValueError('Actual input GT support differs from official support')
        if source['grid_transform']!='identity' or source['canonical_from_world']!=obj['canonical_from_world']:
            raise ValueError('Canonical/grid convention changed')
        frames = [base.FRAMES[i] for i in obj['visible_indices']]
        if inputs['selected_frames']!=frames or len(sample['slat_condition']['cond'])!=len(frames):
            raise ValueError('Observation frames/cameras desynchronized')
        inputs_checked.append({'uid':obj['uid'],'original_index':obj['original_index'],'inputs':inputs,
            'target_voxels':len(target_coords),'gt_coords_equal_sample_and_slat':True,
            'selected_frames':frames,'gt_features_used_for_sampling':False})
        for seed in SEEDS:
            cs = [next(c for c in p['cases'] if case_key(c)==(obj['uid'],seed,kind)) for kind in KINDS]
            ref,gtc = cs
            _,mesh,_,ss = get_result(ref)
            gt = read(verify(gtc['support_record']))
            for row in [ss,gt,mesh]:
                if row['inputs']!=inputs or row['object_uid']!=obj['uid'] or row['seed']!=seed:
                    raise ValueError('Reference/support/condition binding mismatch')
            for field in ['mesh','world_mesh','gt','ss']:
                verify(mesh[field])
            supports = []
            for c,row in zip(cs,[ss,gt]):
                with np.load(verify(row['coords'])) as data:
                    coords = canonical_support(data['coords'])
                metrics = occupancy_metrics(coords,source['target_coords'].numpy())
                if metrics != row['occupancy']:
                    raise ValueError('Support occupancy metrics differ')
                if c['support_kind']=='gt' and not np.array_equal(coords,target_coords):
                    raise ValueError('Oracle coordinates differ from target')
                supports.append(coords)
            expected = dict(zip(['visible','externally_occluded','other_unobserved'],EXPECTED_PARTITIONS[obj['uid']]))
            if partition_counts(mesh)!=expected:
                raise ValueError('Archived fixed GT partition changed')
            codes = [np.ravel_multi_index(c[:,1:].T,(64,64,64)) for c in supports]
            pairs.append({'uid':obj['uid'],'seed':seed,'predicted_voxels':len(supports[0]),
                'gt_voxels':len(supports[1]),'common_coordinates':len(np.intersect1d(*codes)),
                'partition_counts':expected})
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    from .summarize import self_check_statistics
    statistics_check = self_check_statistics()
    write(OUT/'preflight.json',{'time':now(),'identity':ident,'passed':True,
        'input_audits':inputs_checked,'coordinate_audits':pairs,'statistics_check':statistics_check,
        'new_jobs':12,'reference_jobs':12,'cuda_initialized':False,
        'oracle_scope':'GT surface support coordinates only; SLat features remain generated from images/noise'})
    event('prepare_complete',new_meshes=12,referenced_meshes=12,input_caches=4)


if __name__=='__main__':
    main()
