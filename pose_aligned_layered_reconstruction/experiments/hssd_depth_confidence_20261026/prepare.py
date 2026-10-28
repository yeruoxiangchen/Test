"""CPU input/target/reference audit before any crossed evidence or model load."""
import numpy as np
import torch
from .common import *
from ...v2.contracts import validate_sample
from ...v2.dataset import select_views
from ..hssd_cell_query_20261023.model import observations


def main():
    torch.set_num_threads(2)
    identity=freeze_runtime();p=plan()['plan'];rows=[]
    if len(logical_cases())!=60 or len(new_jobs())!=24:raise ValueError('Unexpected factorial denominator')
    for obj in p['objects']:
        cases=[case for case in logical_cases() if case['uid']==obj['uid']]
        conditions=[read(verify(case['condition_record'])) for case in cases]
        inputs=conditions[0]['inputs']
        if any(row['inputs']!=inputs for row in conditions):raise ValueError('Image condition differs between cells')
        source=torch.load(verify(inputs['sample']),map_location='cpu',weights_only=True)
        validate_sample(source,require_labels=False,require_ss_target=True)
        with np.load(verify(obj['ss'])) as target:
            if not np.array_equal(source['target'].numpy(),target['target']):raise ValueError('SS target latent changed')
            if not np.array_equal(source['target_coords'].numpy(),target['official_coords']):raise ValueError('SS support changed')
        if source['object_uid']!=obj['uid'] or source['canonical_from_world']!=obj['canonical_from_world'] or source['grid_transform']!='identity':
            raise ValueError('Wrong cached input or coordinate system')
        selected=obj['visible_indices'];frames=[[0,4,7,11][i] for i in selected]
        sample=select_views(observations(source),torch.tensor(selected,dtype=torch.long))
        if (inputs['selected_indices']!=selected or inputs['selected_frames']!=frames
                or len(sample['intrinsics'])!=len(frames) or len(sample['slat_condition']['cond'])!=len(frames)):
            raise ValueError('Unsynchronized views')
        if any(key in sample for key in ('target','target_coords','target_depth_z_world','layer_depths')):
            raise ValueError('Supervision leaked into observations')
        references=[]
        for case in cases:
            if not case['is_reference']:continue
            _,mesh,_,ss=get_result(case)
            if (mesh['identity']!=case['expected_identity'] or ss['identity']!=case['expected_identity']
                    or mesh['inputs']!=inputs or ss['inputs']!=inputs or mesh['slat_binding']!=p['slat_binding']
                    or mesh['fitted_transform'] or mesh['calls']!=CALLS or ss['calls']!=CALLS
                    or not mesh['success'] or not ss['success']):raise ValueError('Changed historical case')
            counts=tuple(mesh['partitioned'][key]['gt_samples'] for key in ('visible','externally_occluded','other_unobserved'))
            if counts!=EXPECTED_PARTITIONS[obj['uid']]:raise ValueError('GT evaluation partition changed')
            for value in (mesh['mesh'],mesh['world_mesh'],ss['coords']):verify(value)
            references.append(case['reference_mesh'])
        if len(references)!=9:raise ValueError('Missing archived object/seed reference')
        rows.append({'uid':obj['uid'],'frames':frames,'inputs':inputs,'target_shape':list(source['target'].shape),
            'target_voxels':len(source['target_coords']),'references':references,
            'physical_sample':str(verify(inputs['sample']))})
    from .summarize import self_check_statistics
    from .evidence import self_test
    statistics=self_check_statistics();evidence=self_test()
    if torch.cuda.is_initialized():raise ValueError('CPU preparation initialized CUDA')
    write(OUT/'preflight.json',{'time':now(),'identity':identity,'passed':True,'objects':rows,
        'new_arms':list(ARMS),'new_ss':24,'new_meshes':24,'old_references':36,'logical_cases':60,
        'primary_factorial_cases':48,'context_baseline_cases':12,'steps_per_new_arm':200,
        'statistics_check':statistics,'evidence_unit_check':evidence,'cuda_initialized':False,
        'scope':'actual image/target/reference CPU audit; actual eight diagonal evidence replays and CUDA smoke follow'})
    event('prepare_complete',objects=4,logical_cases=60,primary_factorial_cases=48)


if __name__=='__main__':main()
