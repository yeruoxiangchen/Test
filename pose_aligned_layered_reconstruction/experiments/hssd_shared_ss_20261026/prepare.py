"""CPU-only audit of the matched training data and frozen old references."""
import numpy as np
import torch
from .common import *
from .data import load_case


def main():
    torch.set_num_threads(2)
    ident=freeze_runtime();p=plan()['plan'];cache={};rows=[]
    for obj in p['objects']:
        source,sample,inputs,ev,evb=load_case(obj,'appearance_control',cache)
        assert ev is None and evb is None
        with np.load(verify(obj['ss'])) as a:
            if not np.array_equal(a['target'],source['target'].numpy()):
                raise ValueError('Training SS latent changed')
            if not np.array_equal(a['official_coords'],source['target_coords'].numpy()):
                raise ValueError('Target support changed')
        if source['canonical_from_world']!=obj['canonical_from_world'] or source['grid_transform']!='identity':
            raise ValueError('Canonical target/frame changed')
        selected=[base.FRAMES[i] for i in obj['visible_indices']]
        if inputs['selected_frames']!=selected or len(sample['intrinsics'])!=len(selected):
            raise ValueError('Selected views are not synchronized')
        if len(sample['slat_condition']['cond'])!=len(selected):
            raise ValueError('SLat view selection differs')
        if any(key in sample for key in ['target','target_coords','layer_depths','target_depth_z_world']):
            raise ValueError('Supervision leaked into image observation sample')
        old=[]
        for seed in SEEDS:
            case=next(c for c in logical_cases() if case_key(c)==('frozen',obj['uid'],seed))
            _,mesh,_,ss=get_result(case)
            if mesh['inputs']!=inputs or ss['inputs']!=inputs or mesh['slat_binding']!=p['slat_binding']:
                raise ValueError('Old reference observations/model differ')
            if mesh['fitted_transform'] or mesh['calls']!=CALLS or not mesh['success'] or not ss['success']:
                raise ValueError('Invalid archived reference')
            counts=tuple(mesh['partitioned'][k]['gt_samples'] for k in ['visible','externally_occluded','other_unobserved'])
            if counts!=EXPECTED_PARTITIONS[obj['uid']]:raise ValueError('GT partitions differ')
            for b in [mesh['mesh'],mesh['world_mesh'],ss['coords']]:verify(b)
            old.append(case['reference_mesh'])
        rows.append({'uid':obj['uid'],'original_index':obj['original_index'],'inputs':inputs,
            'frames':selected,'target_shape':list(source['target'].shape),'target_voxels':len(source['target_coords']),
            'oracle_geometry_not_in_sample':True,'references':old})
    from .summarize import self_check_statistics
    stats=self_check_statistics()
    if torch.cuda.is_initialized():raise ValueError('CPU preparation initialized CUDA')
    write(OUT/'preflight.json',{'time':now(),'identity':ident,'passed':True,'objects':rows,
        'new_ss':24,'new_meshes':24,'old_references':12,'logical_cases':36,'steps_per_arm':STEPS,
        'statistics_check':stats,'cuda_initialized':False,
        'scope':'source/target checks only; depth/evidence and real training smoke follow'})
    event('prepare_complete',objects=4,steps_per_arm=STEPS)


if __name__=='__main__':main()
