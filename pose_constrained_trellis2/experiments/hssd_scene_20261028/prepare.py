"""Freeze 8 training + 4 validation instances before native codec scores."""
import itertools
import numpy as np
from . import common as c

def main():
    source=c.read(c.SOURCE/'targets.json');rows=source['plan']['objects']
    chosen=[];groups=[]
    for subset,count in [('train',2),('val',1)]:
        scenes=sorted({r['scene'] for r in rows if r['split']==subset},key=lambda s:c.digest(['scene_native_v1',s]))[:count]
        for sid in scenes:
            pool=sorted([r for r in rows if r['scene']==sid],key=lambda r:c.digest(['scene_native_v1',r['uid']]))
            anchor=pool[0];center=np.linalg.inv(anchor['canonical_from_world'])[:3,3]
            pool.sort(key=lambda r:(float(np.linalg.norm(np.linalg.inv(r['canonical_from_world'])[:3,3]-center)),r['uid']))
            group=pool[:4];chosen.extend(group)
            groups.append({'scene':sid,'split':subset,'uids':[r['uid'] for r in group],
                'pairs':[{'a':a['uid'],'b':b['uid'],
                         'selected_common_frames':sorted({v['frame'] for v in a['views']}&{v['frame'] for v in b['views']}),
                         'distance_m':float(np.linalg.norm(np.linalg.inv(a['canonical_from_world'])[:3,3]-np.linalg.inv(b['canonical_from_world'])[:3,3]))}
                         for a,b in itertools.combinations(group,2)]})
    if len(chosen)!=12 or len({r['asset_family'] for r in chosen})!=12:raise ValueError('Invalid pilot cohort')
    payload={'source_targets_identity':source['identity'],'objects':chosen,'groups':groups,
      'scope':'native codec + full sampling; known camera and object transforms; not RGB-only deployment',
      'selection':'fixed scene hash, fixed anchor hash, nearest three target centers; before codec/model scores',
      'sampling_seeds':[42],'training_smoke_steps':3,'formal_training_launched':False,
      'shape_resolution':512,'ss_support_source':'native Shape-1024 coords at64',
      'failure_policy':'retain all 12; stop on invalid codec, never substitute targets'}
    c.immutable(c.OUT/'plan.json',{'identity':c.digest(payload),'plan':payload})
    for row in chosen:
        if c.sha(row['mesh']['path'])!=row['mesh']['sha256']:raise ValueError('Changed canonical mesh')
    c.write(c.OUT/'selection.json',{'time':c.now(),'train':8,'val':4,'scenes':3,'groups':groups})
    print('Frozen native pilot: 8 train + 4 val, 3 scene groups',flush=True)

if __name__=='__main__':main()
