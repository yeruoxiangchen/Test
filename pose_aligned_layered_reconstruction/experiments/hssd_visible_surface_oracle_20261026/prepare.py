"""CPU audit of unchanged inputs/targets and complete historical references."""
import numpy as np
import torch
from .common import *
from ...v2.contracts import validate_sample
from ...v2.dataset import select_views
from ..hssd_cell_query_20261023.model import observations


def main():
    torch.set_num_threads(2)
    ident=freeze_runtime();p=plan()['plan'];rows=[]
    if len(logical_cases())!=36 or len(new_jobs())!=12:raise ValueError('Unexpected cohort size')
    for obj in p['objects']:
        cases=[c for c in logical_cases() if c['uid']==obj['uid']]
        records=[read(verify(c['condition_record'])) for c in cases]
        inputs=records[0]['inputs']
        if any(r['inputs']!=inputs for r in records):raise ValueError('Different image inputs between cases')
        value=torch.load(verify(inputs['sample']),map_location='cpu',weights_only=True)
        validate_sample(value,require_labels=False,require_ss_target=True)
        if value['object_uid']!=obj['uid']:raise ValueError('Wrong cached object')
        with np.load(verify(obj['ss'])) as a:
            if not np.array_equal(a['target'],value['target'].numpy()):raise ValueError('SS latent differs')
            if not np.array_equal(a['official_coords'],value['target_coords'].numpy()):raise ValueError('SS support differs')
        if value['canonical_from_world']!=obj['canonical_from_world'] or value['grid_transform']!='identity':
            raise ValueError('Canonical frame differs')
        selected=obj['visible_indices'];frames=[[0,4,7,11][i] for i in selected]
        sample=select_views(observations(value),torch.tensor(selected,dtype=torch.long))
        if (inputs['selected_indices']!=selected or inputs['selected_frames']!=frames
                or len(sample['intrinsics'])!=len(frames) or len(sample['slat_condition']['cond'])!=len(frames)):
            raise ValueError('View selection not synchronized')
        if any(k in sample for k in ('target','target_coords','layer_depths','target_depth_z_world')):
            raise ValueError('Supervision leaked into observations')
        c=np.asarray(obj['canonical_from_world']);inverse=np.linalg.inv(c)
        if not np.allclose(c@inverse,np.eye(4),rtol=0,atol=1e-12):raise ValueError('Invalid canonical mapping')
        refs=[]
        for case in cases:
            if not case['is_reference']:continue
            _,mesh,_,ss=get_result(case)
            if (mesh['inputs']!=inputs or ss['inputs']!=inputs or mesh['slat_binding']!=p['slat_binding']
                    or mesh['fitted_transform'] or mesh['calls']!=CALLS or ss['calls']!=CALLS
                    or not mesh['success'] or not ss['success']):raise ValueError('Invalid reference')
            counts=tuple(mesh['partitioned'][k]['gt_samples'] for k in ('visible','externally_occluded','other_unobserved'))
            if counts!=EXPECTED_PARTITIONS[obj['uid']]:raise ValueError('GT partitions differ')
            for b in (mesh['mesh'],mesh['world_mesh'],ss['coords']):verify(b)
            refs.append(case['reference_mesh'])
        if len(refs)!=6:raise ValueError('Reference denominator changed')
        rows.append({'uid':obj['uid'],'frames':frames,'target_shape':list(value['target'].shape),
            'target_voxels':len(value['target_coords']),'input':inputs,'references':refs,
            'payload_physical_path':str(verify(inputs['sample'])),'source_files_policy':'bound provenance only; no raw re-encoding'})
    for item in mappings().values():
        if not str(Path(item['physical']).resolve()).startswith('/data/zjr/'):
            raise ValueError('Payload staging escaped /data/zjr')
    from .summarize import self_check_statistics
    from .evidence import self_test
    stats=self_check_statistics();evidence=self_test()
    if torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
    write(OUT/'preflight.json',{'time':now(),'identity':ident,'passed':True,'objects':rows,
        'new_ss':12,'new_meshes':12,'old_references':24,'logical_cases':36,'steps_per_arm':STEPS,
        'statistics_check':stats,'evidence_unit_check':evidence,'cuda_initialized':False,
        'scope':'source/target/CPU invariants only; real model smoke remains required'})
    event('prepare_complete',objects=4,new_meshes=12,references=24)


if __name__=='__main__':main()
