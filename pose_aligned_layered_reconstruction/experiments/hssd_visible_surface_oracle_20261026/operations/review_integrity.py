"""CPU artifact, provenance and coordinate audit; never changes frozen results.

Default prints a short JSON result without writing. --write publishes the audit
under /data and must be launched with the appropriate filesystem permission.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=Path('/data/zjr/pose_aligned_layered_reconstruction')
for key,suffix in {'TMPDIR':'tmp','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface',
    'HF_HUB_CACHE':'cache/huggingface/hub','TORCH_EXTENSIONS_DIR':'cache/torch_extensions',
    'MPLCONFIGDIR':'cache/matplotlib','NUMBA_CACHE_DIR':'cache/numba','XDG_CACHE_HOME':'cache/xdg'}.items():
    os.environ[key]=str(DATA/suffix)
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MKL_THREADING_LAYER']='GNU'
sys.path.insert(0,str(ROOT))
c=importlib.import_module('pose_aligned_layered_reconstruction.experiments.hssd_visible_surface_oracle_20261026.common')

import numpy as np
import torch
import trimesh

torch.set_num_threads(2)
LIMIT=256*1024*1024


class Audit:
    def __init__(self):
        self.checks=[];self.hashed={};self.skipped={};self.json_seen=set()

    def check(self,name,value,**fields):
        if not bool(value):raise AssertionError(name+': '+json.dumps(fields,ensure_ascii=False))
        self.checks.append({'check':name,'passed':True,**fields})

    def bound(self,b):
        path=c.physical(b['path']);key=(b['path'],b['sha256'])
        stat=path.stat()
        if stat.st_size>LIMIT:
            self.check('large_weight_physical_storage',str(path).startswith('/data/zjr/'),path=str(path))
            self.skipped[key]={'logical':b['path'],'physical':str(path),'bytes':stat.st_size,
                'expected_sha256':b['sha256'],'reason':'large model already verified at staging/load; not byte-rehashed in this review'}
            return path
        c.verify(b)
        self.hashed[key]={'logical':b['path'],'physical':str(path),'bytes':stat.st_size,'sha256':b['sha256']}
        return path

    def tree(self,value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value and isinstance(value['path'],str):self.bound(value)
            if 'checkpoint' in value and 'sha256' in value and isinstance(value['checkpoint'],str):
                self.bound({'path':value['checkpoint'],'sha256':value['sha256']})
            for key,item in value.items():
                if key not in ('path','sha256'):self.tree(item)
        elif isinstance(value,list):
            for item in value:self.tree(item)

    def load(self,b):return c.read(self.bound(b))


def array_hash(value):
    arr=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
    return h.hexdigest()


def tensor_receipt(value):
    value=value.detach().contiguous().cpu()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def main(write=False):
    a=Audit();out=c.OUT
    plan=c.plan();p=plan['plan'];summary=c.read(out/'summary.json');runtime=c.read(out/'evaluation_identity.json')
    identity=c.digest(runtime)
    a.check('runtime_identity',identity==summary['identity'] and runtime['plan_identity']==plan['identity'])
    a.check('completed_case_counts',(summary['new_meshes'],summary['new_ss_samples'],summary['original_references'],summary['logical_cases'])==(12,12,24,36))
    a.check('case_cartesian_product',len(p['cases'])==36 and
        {(x['arm'],x['uid'],x['seed']) for x in p['cases']}==
        {(arm,uid,seed) for arm in c.ALL_ARMS for uid in c.UIDS for seed in c.SEEDS})
    a.tree(runtime);a.tree(summary['plan']);a.tree(p['storage_manifest'])
    storage=a.load(p['storage_manifest'])
    stock=[(logical,row) for logical,row in storage['mapping'].items()
        if '/models--Stable-X--trellis-vggt-v0-2/' in logical]
    a.check('four_stock_models_staged',len(stock)==8)
    for logical,row in stock:
        a.bound({'path':logical,'sha256':row['sha256']})
    # Explicitly revalidate all frozen local/inherited numerical sources.
    for source in runtime['sources']+runtime['inherited_sources']:a.bound(source)
    actual=[c.binding(path) for path in sorted(c.HERE.glob('*.py'))+[c.HERE/'run.sh',c.HERE/'PROTOCOL.md']]
    a.check('frozen_source_inventory',runtime['sources']==actual)
    a.check('previous_runtime_identity',c.digest(a.load(p['previous_evaluation_identity']))==p['previous_identity'])
    for key in ('previous_summary','previous_plan','previous_visual_audit','parent_checkpoint','parent_training_record',
                'shared_plan','runtime_amendment'):
        a.bound(p[key])
    parent_run=a.load(p['parent_training_record']);a.bound(parent_run['contract']['appearance_base'])
    a.check('parent_is_complete_old_appearance',parent_run['contract']['arm']=='appearance_control'
        and parent_run['contract']['steps']==200)

    smoke=c.read(out/'smoke.json');a.tree(smoke)
    a.check('smoke_identity_and_complete',smoke['identity']==identity and all(smoke[k] is True for k in
        ('passed','parent_ss_mesh_replay_exact','zero_initialization_exact','matched_trainable_capacity',
         'two_step_training_passed','shared_noise_order_exact')))
    for key in ('initial_check','training_check','replay'):
        receipt=a.load(smoke[key]);a.tree(receipt);a.check('smoke_receipt_'+key,receipt['passed'] is True)
    complete=c.read(out/'train/geometry_oracle_visible/complete.json')
    run=c.read(out/'train/geometry_oracle_visible/run.json');a.tree(run);a.tree(complete)
    checkpoint=torch.load(a.bound(complete['checkpoint']),map_location='cpu',weights_only=False)
    a.check('trained_checkpoint_cursor',checkpoint['step']==complete['step']==200)
    a.check('trained_checkpoint_contract',checkpoint['contract']==run['contract']==complete['contract'])
    a.check('trained_checkpoint_data',checkpoint['data_identity']==run['data_identity']==complete['data_identity'])
    a.check('trained_checkpoint_residual',c.tensor_hash(checkpoint['residual'])==complete['final_residual_sha'])
    a.check('frozen_parent_versions',checkpoint.get('frozen_parameter_versions_unchanged') is True)
    a.check('initialization_and_capacity',checkpoint['initial_residual_sha']==run['initial_residual_sha']
        and run['trainable_parameter_count']==3164160)
    del checkpoint
    for arm,rows in p['reference_training'].items():
        for b in rows.values():a.bound(b)

    source_inputs={};target_hashes={}
    for obj in p['objects']:
        row=next(x for x in run['data_identity'] if x['uid']==obj['uid'])
        value=torch.load(a.bound(row['inputs']['sample']),map_location='cpu',weights_only=True)
        with np.load(a.bound(obj['ss'])) as target:
            a.check('actual_ss_target_'+obj['uid'],np.array_equal(value['target'].numpy(),target['target'])
                and np.array_equal(value['target_coords'].numpy(),target['official_coords']))
        a.check('actual_target_tensor_hash_'+obj['uid'],c.tensor_hash({'target':value['target']})==row['target_tensor_sha'])
        a.check('input_native_frame_'+obj['uid'],value['canonical_from_world']==obj['canonical_from_world']
            and value['grid_transform']=='identity')
        source_inputs[obj['uid']]=row['inputs'];target_hashes[obj['uid']]=row['target_tensor_sha']
        a.bound(obj['mesh']);a.bound(obj['slat'])

    observations=c.read(out/'observations/complete.json');a.tree(observations)
    a.check('stripped_observation_complete',observations['success'] is True and observations['frames']==[0,4,7,11]
        and observations['hidden_geometry_used'] is False)
    for b in observations['receipts']:
        row=a.load(b);a.tree(row)
        with np.load(a.bound(row['npz'])) as data:
            a.check('observation_exact_allowlist',set(data.files)=={'depth_z_world','valid'})
            depth,valid=data['depth_z_world'],data['valid']
            a.check('observation_validity',depth.dtype==np.float32 and valid.dtype==np.bool_ and
                depth.shape==(518,518) and valid.shape==depth.shape and np.isfinite(depth).all()
                and np.all(depth[valid]>0) and np.all(depth[~valid]==0))
    evidence=c.read(out/'evidence/complete.json');a.tree(evidence)
    a.check('evidence_observation_scope',len(evidence['objects'])==4 and evidence['hidden_geometry_used'] is False
        and evidence['hard_carving'] is False and evidence['confidence_error_removed'] is True
        and evidence['depth_only_ablation'] is False)
    for b in evidence['objects']:
        row=a.load(b);a.tree(row)
        expected=[0,11] if row['uid'].endswith('i065') else [0,4,7,11]
        a.check('evidence_selected_frames_'+row['uid'],row['sources']['selected_frames']==expected)

    records={};hashes={};coordinate_rows=[]
    for case in p['cases']:
        key=case['arm'],case['uid'],case['seed'];obj=next(o for o in p['objects'] if o['uid']==case['uid'])
        r=next(r for r in summary['records'] if (r['arm'],r['uid'],r['seed'])==key)
        mesh=a.load(r['mesh_record']);ss=a.load(r['ss_record']);condition=a.load(case['condition_record'])
        a.tree(mesh);a.tree(ss)
        a.check('embedded_raw_mesh_'+str(key),r['mesh']==mesh)
        expected_identity=p['previous_identity'] if case['is_reference'] else identity
        a.check('case_runtime_'+str(key),mesh['identity']==ss['identity']==expected_identity)
        a.check('case_success_'+str(key),mesh['success'] is True and ss['success'] is True)
        a.check('case_inputs_'+str(key),mesh['inputs']==ss['inputs']==condition['inputs']==source_inputs[case['uid']])
        a.check('case_geometry_contract_'+str(key),mesh['canonical_from_world']==obj['canonical_from_world']
            and mesh['gt']==obj['mesh'] and mesh['fitted_transform'] is False)
        a.check('case_slat_calls_'+str(key),mesh['slat_binding']==p['slat_binding'] and mesh['calls']==ss['calls']==c.CALLS)
        a.check('case_partition_counts_'+str(key),tuple(mesh['partitioned'][k]['gt_samples'] for k in
            ('visible','externally_occluded','other_unobserved'))==c.EXPECTED_PARTITIONS[case['uid']])
        with np.load(a.bound(ss['coords'])) as data:coords=data['coords']
        a.check('case_support_'+str(key),coords.dtype==np.int32 and coords.ndim==2 and coords.shape[1]==4 and len(coords)>0
            and np.all(coords[:,0]==0) and np.all((coords[:,1:]>=0)&(coords[:,1:]<64))
            and np.array_equal(np.unique(coords,axis=0),coords) and len(coords)==ss['occupancy']['predicted_voxels'])
        lr=a.load(mesh['sampled_slat_receipt']);a.tree(lr)
        latent=torch.load(a.bound(lr['latent']),map_location='cpu',weights_only=True)
        a.check('case_latent_support_'+str(key),np.array_equal(latent['coords'].numpy(),coords)
            and latent['feats'].shape==(len(coords),8) and torch.isfinite(latent['feats']).all())
        a.check('case_latent_contract_'+str(key),lr['contract']['inputs']==ss['inputs']
            and lr['contract']['slat_binding']==p['slat_binding'] and lr['calls']==c.CALLS
            and lr['contract']['master_seed']==case['seed']*2000003+obj['original_index']*2017+7919)
        a.check('case_initial_coord_hash_'+str(key),tensor_receipt(latent['coords'])==lr['contract']['initial_coords'])
        canonical=trimesh.load(a.bound(mesh['mesh']),process=False,force='mesh')
        a.check('case_mesh_structure_'+str(key),len(canonical.vertices)==mesh['structure']['vertex_count']
            and len(canonical.faces)==mesh['structure']['face_count'] and np.isfinite(canonical.vertices).all())
        hashes[key]={'coords':array_hash(coords),'latent':tensor_receipt(latent['feats'])['sha256'],
            'mesh':array_hash(canonical.vertices)+':'+array_hash(canonical.faces)}
        if not case['is_reference']:
            world=trimesh.load(a.bound(mesh['world_mesh']),process=False,force='mesh')
            cfw=np.asarray(obj['canonical_from_world'],np.float64);wfc=np.linalg.inv(cfw)
            expected=canonical.vertices@wfc[:3,:3].T+wfc[:3,3]
            world_error=float(np.max(np.abs(world.vertices-expected)))
            roundtrip=world.vertices@cfw[:3,:3].T+cfw[:3,3]
            roundtrip_error=float(np.max(np.abs(roundtrip-canonical.vertices)))
            a.check('new_mesh_full_coordinate_mapping_'+str(key),np.array_equal(world.faces,canonical.faces)
                and world_error<=1e-6 and roundtrip_error<=1e-6)
            a.check('new_mesh_export_error_matches_'+str(key),abs(world_error-mesh['world_export_max_abs_error'])<=1e-12)
            coordinate_rows.append({'uid':case['uid'],'seed':case['seed'],'vertices':len(canonical.vertices),
                'faces':len(canonical.faces),'world_max_abs_error_m':world_error,
                'canonical_roundtrip_max_abs_error':roundtrip_error,'faces_equal':True,'fitted_transform':False})
        records[key]=(mesh,ss)
    for uid in c.UIDS:
        for field in ('coords','latent','mesh'):
            vals=[hashes['geometry_oracle_visible',uid,seed][field] for seed in c.SEEDS]
            a.check('new_seed_independence_'+uid+'_'+field,len(set(vals))==3)
        for seed in c.SEEDS:
            for old in c.ALL_ARMS[:2]:
                a.check('new_vs_reference_arrays_differ_'+uid+'_'+str(seed)+'_'+old,
                    all(hashes['geometry_oracle_visible',uid,seed][field]!=hashes[old,uid,seed][field]
                        for field in ('coords','latent','mesh')))

    visual=a.load(summary['visual_audit']);a.tree(visual)
    a.check('figure_world_counts',len(visual['figures'])==len(summary['figures'])==12
        and len(visual['worlds'])==len(summary['worlds'])==9 and len(visual['render_records'])==40)
    a.check('visual_native_camera_policy',not visual['clipping'] and visual['human_review_completed'] is False)
    for b in visual['figures']:a.bound(b)
    for b in visual['worlds']:a.tree(b);a.check('world_four_objects',b['objects']==4 and len(b['sources'])==4)
    for render_row in visual['render_records']:
        row=a.load(render_row['record']);a.tree(row)
        a.check('native_render_transform',np.array_equal(np.asarray(row['matrix_original_to_rendered']),np.eye(4))
            and len(row['views'])==3)
        for v in row['views']:
            a.bound({'path':v['png'],'sha256':v['png_sha256']})
            a.check('render_unclipped',v['vertices_outside_xy']==v['vertices_outside_depth']==0
                and v['foreground_touches_image_border'] is False)
    for op in sorted((out/'operations').glob('gpu_resume_*.json')):
        row=c.read(op);a.tree(row)
        a.check('operational_handoff_kept_numeric_identity',row['runtime_identity']==identity and row['numeric_runtime_unchanged'])
    a.check('cpu_only',not torch.cuda.is_initialized())
    result={'time':c.now(),'passed':True,'runtime_identity':identity,'plan_identity':plan['identity'],
        'checked_cases':36,'new_cases':12,'reference_cases':24,'checks_passed':len(a.checks),
        'file_hashes_verified':len(a.hashed),'large_bindings_not_rehashed':list(a.skipped.values()),
        'source_files_verified':sum(Path(r['logical']).suffix in ('.py','.sh','.md','.cpp','.cu','.h') for r in a.hashed.values()),
        'coordinate_audit':{'new_mesh_pairs':12,'vertices_checked':sum(r['vertices'] for r in coordinate_rows),
            'world_max_abs_error_m':max(r['world_max_abs_error_m'] for r in coordinate_rows),
            'canonical_roundtrip_max_abs_error':max(r['canonical_roundtrip_max_abs_error'] for r in coordinate_rows),
            'rows':coordinate_rows},
        'actual_array_independence':{'new_seeds_independent':True,'new_vs_both_references_differ':True,
            'all_36_array_hashes':[{'arm':k[0],'uid':k[1],'seed':k[2],**v} for k,v in hashes.items()]},
        'checks':a.checks,'verified_files':list(a.hashed.values()),'cuda_initialized':False,
        'limits':['No GPU regeneration or surface resampling; recorded surface metrics are not recomputed here.',
            'World-coordinate vertex arrays audited for all 12 new meshes; old reference transforms were audited previously.',
            'Large stock/30K weights not rehashed; frozen bindings and staged paths checked.',
            'Raw original scenes and old VGGT predictor are not reopened; reuse is tied to bound input samples/results.',
            'Visual-quality interpretation belongs to separate image review; zero clipping is an artifact check.']}
    if write:c.write(out/'review_20261026/verification.json',result)
    print(json.dumps({k:result[k] for k in ('passed','runtime_identity','checks_passed','file_hashes_verified','source_files_verified',
        'coordinate_audit','large_bindings_not_rehashed')},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();main(args.write)
