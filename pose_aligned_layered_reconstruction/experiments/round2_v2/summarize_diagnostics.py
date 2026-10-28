"""Refuse incomplete pairs; report object-clustered effects and native mesh metrics."""
import argparse
from collections import defaultdict
import json
import numpy as np
from .diagnostic_suite import OUT, ANALYSIS, ARMS, cohort_indices, atomic_json, verify


def pair(rows, after, before, metric, sign=1):
    lookup={(r['index'],r['seed'],r['mode']):r for r in rows}
    grouped=defaultdict(list)
    for r in rows:
        if r['mode']==after:
            previous=lookup[r['index'],r['seed'],before]
            grouped[r['object_uid']].append((r['index'], r['seed'], sign*(r[metric]-previous[metric])))
    # Canonical object and case ordering makes bootstrap draws independent of worker/file creation order.
    means={k:float(np.mean([v[2] for v in sorted(grouped[k])])) for k in sorted(grouped)}
    a=np.array(list(means.values()));rng=np.random.default_rng(20261018)
    draws=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
    return {'object_count':len(a),'mean_improvement':float(a.mean()),'positive_means_better':True,
            'wins':int((a>1.e-12).sum()),'ties':int((np.abs(a)<=1.e-12).sum()),
            'losses':int((a< -1.e-12).sum()),'object_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(),
            'per_object':means}


def check(rows, indices, modes):
    keys=[(r['index'],r['seed'],r['mode']) for r in rows]
    expected={(i,s,m) for i in indices for s in (42,43) for m in modes}
    if len(set(keys))!=len(keys) or set(keys)!=expected:
        raise ValueError(f'incomplete/duplicate cohort: {len(keys)} records, expected {len(expected)}')
    uids={}
    for r in rows:
        if uids.setdefault(r['index'],r['object_uid'])!=r['object_uid']:
            raise ValueError('paired object identity mismatch')


def read_ss(label, modes, indices):
    rows=[]
    for mode in modes:
        folder=OUT/label/mode
        identities=[json.loads(p.read_text())['identity_hash'] for p in folder.glob('identity*.json')]
        if not identities or len(set(identities))!=1:raise ValueError('mixed/missing SS identities: '+mode)
        for path in (folder/'records').glob('*.json'):
            r=json.loads(path.read_text())
            if r['identity_hash']!=identities[0]:raise ValueError('SS record hash mismatch')
            if not all(np.isfinite(r[k]) for k in ('iou','precision','recall','count_ratio')):
                raise ValueError('nonfinite SS metric')
            verify(r['coordinates']);rows.append(r)
    check(rows,indices,modes)
    return rows


def ss_report(label):
    if label=='pilot32':
        indices=cohort_indices('paired32',96);modes=['baseline500',*ARMS]
        rows=read_ss(label,modes,indices)
        previous=json.loads((ANALYSIS/'occupancy_report.json').read_text())['records']
        rows.extend(r for r in previous if r['index'] in indices and r['mode']=='parent')
        modes=['parent',*modes];check(rows,indices,modes)
        comparisons=[(m,'parent') for m in modes[1:]] + [('raw_joint','baseline500'),
            ('raw_frozen','raw_joint'),('raw_frozen_v1recipe','raw_frozen'),
            ('raw_frozen_v1recipe_tol004','raw_frozen_v1recipe')]
        groups={'all':indices,'clean':list(range(32)),'occluded':list(range(32,96,2))}
    else:
        indices=list(range(16 if label=='strong16' else 64));modes=['parent','v1','raw']
        if label=='dev64' and (OUT/'dev64/bias_only/complete.json').exists():modes.append('bias_only')
        if label=='dev64' and (OUT/'dev64/layer_view_roll/complete.json').exists():modes.append('layer_view_roll')
        rows=read_ss(label,modes,indices)
        comparisons=[('v1','parent'),('raw','parent'),('raw','v1')]
        if 'bias_only' in modes:comparisons += [('bias_only','parent'),('raw','bias_only')]
        if 'layer_view_roll' in modes:comparisons += [('layer_view_roll','parent'),('raw','layer_view_roll')]
        groups={'all':indices}
    result={}
    for name, subset in groups.items():
        rr=[r for r in rows if r['index'] in subset]
        metrics={mode:{k:float(np.mean([r[k] for r in rr if r['mode']==mode]))
                       for k in ('iou','precision','recall','count_ratio')} for mode in modes}
        for mode in modes:metrics[mode]['empty_count']=sum(r['predicted_voxels']==0 for r in rr if r['mode']==mode)
        result[name]={'metrics':metrics,'paired':{f'{a}_minus_{b}':pair(rr,a,b,'iou') for a,b in comparisons}}
    atomic_json(OUT/label/'aggregate.json',{'groups':result,'record_count':len(rows),
                'inference_seeds':[42,43],'training_seed_count':1,'complete_pairing':True})
    print(label,json.dumps({k:v['metrics'] for k,v in result.items()}))


def mesh_report():
    payload=json.loads((OUT/'mesh/plan.json').read_text());plan=payload['plan']
    rows=[]
    for path in (OUT/'mesh/records').glob('*.json'):
        r=json.loads(path.read_text())
        if r['identity_hash']!=payload['identity_hash'] or not r['success']:
            raise ValueError('failed/mixed mesh evaluation')
        verify(r['mesh']);rows.append({**r,**r['surface']})
    indices=[r['index'] for r in plan['rows']]
    check(rows,indices,plan['modes'])
    groups={}
    for name,subset in [('all',indices),('clean',list(range(32))),('occluded',list(range(32,96,2)))]:
        rr=[r for r in rows if r['index'] in subset]
        metrics={m:{k:float(np.mean([r[k] for r in rr if r['mode']==m])) for k in
                    ('chamfer_l1','fscore_0p02','precision_0p02','recall_0p02','normal_consistency')} for m in plan['modes']}
        paired={}
        for a,b in [('raw','parent'),('v1','parent'),('raw','v1')]:
            paired[a+'_minus_'+b]={k:pair(rr,a,b,k,sign) for k,sign in [('chamfer_l1',-1),('fscore_0p02',1)]}
        groups[name]={'metrics':metrics,'paired':paired}
    atomic_json(OUT/'mesh/aggregate.json',{'groups':groups,'record_count':len(rows),'complete_pairing':True,
        'scope':'training objects seen by SLat30K; native frame; official VAE proxy GT; no alignment'})
    print('mesh',json.dumps({k:v['metrics'] for k,v in groups.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',choices=['pilot32','dev64','strong16','mesh']);a=p.parse_args()
    mesh_report() if a.label=='mesh' else ss_report(a.label)
