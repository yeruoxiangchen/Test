"""CPU guards for leakage, storage, GPU admission, metadata and transform math."""
import argparse
from collections import defaultdict
import csv
import importlib.util
from pathlib import Path

import numpy as np
from . import common as c
from .split import family_table,split_scenes


def run():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError):check(name,True)
        else:raise AssertionError('Did not reject '+name)
    for bad in ('/home/zjr/Tracker/data','/tmp/data',str(c.DATA/'../other')):
        rejects('deny_output_'+bad,lambda bad=bad:c.data_path(bad))
    check('data_root_accepted',c.data_path(c.DATASET).is_relative_to(c.DATA))
    table,family=family_table([{'id':'a','decomposedFrom':'b'},{'id':'b','decomposedFrom':'c'},{'id':'c','decomposedFrom':''}])
    check('recursive_decomposition',family('a')==family('b')==family('c')=='c')
    _,cycle=family_table([{'id':'a','decomposedFrom':'b'},{'id':'b','decomposedFrom':'a'}])
    rejects('decomposition_cycle',lambda:cycle('a'))
    official={'train':[str(x) for x in range(30)],'val':[str(x) for x in range(30,50)]}
    scene_split=split_scenes(official,set(map(str,range(50))))
    check('scene_counts',{k:len(v) for k,v in scene_split.items()}==c.SCENE_COUNTS)
    check('official_boundary',set(scene_split['train'])<=set(official['train']) and
          set(scene_split['val']+scene_split['test'])<=set(official['val']))
    check('scene_groups_disjoint',not(set(scene_split['train'])&set(scene_split['val']) or
                                     set(scene_split['train'])&set(scene_split['test']) or set(scene_split['val'])&set(scene_split['test'])))
    rejects('official_overlap',lambda:split_scenes({'train':['1']*30,'val':['1']*20},{'1'}))
    spec=importlib.util.spec_from_file_location('_hssd192_runner_check',c.HERE/'runner.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=[{'gpu':'0','used_mib':5000,'free_mib':19000,'compute_pids':[8]},
          {'gpu':'2','used_mib':1,'free_mib':24000,'compute_pids':[]},
          {'gpu':'3','used_mib':1000,'free_mib':23000,'compute_pids':[9]}]
    check('prefer_idle_gpu',runner.choose_gpu(rows,'auto',['0','2','3'])['gpu']=='2')
    check('shared_headroom_gpu',runner.choose_gpu(rows,'auto',['0','3'])['gpu']=='3')
    check('fixed_gpu_honored',runner.choose_gpu(rows,'0',['0','2'])['gpu']=='0')
    check('no_gpu_when_unsafe',runner.choose_gpu([{'gpu':'0','used_mib':12000,'free_mib':12000,'compute_pids':[9]}],'auto',['0']) is None)
    # Round-trip a reflected, nonuniform target while keeping the camera's
    # canonical transform a positive-scale proper rotation.
    from ..hssd_pilot_20261020.assets import instance_matrix,C
    instance={'rotation':[.9238795325,0,.3826834324,0],'translation':[3,1,-2],'non_uniform_scale':[-2,.7,1.3]}
    rigid,deform=instance_matrix(instance);norm=np.diag([.3,.3,.3,1]);norm[:3,3]=[.1,-.05,.02]
    cfw=norm@C@np.linalg.inv(rigid);points=np.array([[.1,.2,.3],[-.2,.1,.4],[.3,-.1,.2]])
    target=points@(norm@C@deform)[:3,:3].T+(norm@C@deform)[:3,3]
    recovered=target@np.linalg.inv(cfw)[:3,:3].T+np.linalg.inv(cfw)[:3,3]
    expected=points@(rigid@deform)[:3,:3].T+(rigid@deform)[:3,3]
    check('reflected_nonuniform_target_roundtrip',np.max(np.abs(recovered-expected))<1e-12)
    rotation=cfw[:3,:3]/.3
    check('camera_rotation_proper',np.max(np.abs(rotation@rotation.T-np.eye(3)))<1e-12 and abs(np.linalg.det(rotation)-1)<1e-12)
    # Known PLY/GLTF recipe remains exactly the prior source; no CUDA touched.
    if (c.OUT/'candidate_plan.json').exists():
        p=c.plan()['plan'];c.verify(p['official_split']);c.verify(p['metadata']);check('actual_24_scenes',sum(map(len,p['scene_splits'].values()))==24)
        check('actual_history_scenes_excluded',not(c.DEVELOPMENT_SCENES&{s for ids in p['scene_splits'].values() for s in ids}))
        _,family=family_table(list(csv.DictReader((c.RAW/'metadata/fpmodels-with-decomposed.csv').open())))
        presence=defaultdict(set);content=defaultdict(set)
        deps={r['scene']:r for r in p['dependencies']['scenes']};files={r['path']:r for r in p['dependencies']['geometry_files']}
        for subset,ids in p['scene_splits'].items():
            for sid in ids:
                cfg={Path(x).name.removesuffix('.object_config.json'):x for x in deps[sid]['configs']}
                for obj in c.read(c.RAW/f'scenes/{sid}.scene_instance.json')['object_instances']:
                    name=obj['template_name'];presence[family(name)].add(subset)
                    remote=files[deps[sid]['render_assets'][cfg[name]]];content[remote.get('sha256') or remote['git_oid']].add(subset)
        check('all_actual_candidate_families_unseen_in_other_contexts',all(presence[r['asset_family']]=={r['split']} for r in p['candidates']))
        check('all_actual_candidate_bytes_unseen_in_other_contexts',all(content[r['source_content_id']]=={r['split']} for r in p['candidates']))
        check('all_actual_candidates_not_historical',all(r['asset_family'] not in p['historical_families'] for r in p['candidates']))
        check('raw_download_budget',p['dependencies']['download_bytes']<=40*1024**3)
    import torch
    check('cpu_did_not_initialize_cuda',not torch.cuda.is_initialized())
    return {'time':c.now(),'passed':True,'checks':checks,'count':len(checks),'source':c.binding(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    value=run()
    if args.write:c.write(c.OUT/'preflight.json',value)
    print(value)
