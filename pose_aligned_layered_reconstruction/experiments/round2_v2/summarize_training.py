"""Audit completed pilot budgets, fixed geometry, and paired schedule prefixes."""
import json
import numpy as np
import torch
from .diagnostic_suite import OUT, SOURCE, ARMS, binding, atomic_json


def main():
    reference=torch.load(SOURCE/'lari/checkpoints/step_002000.pt',map_location='cpu',weights_only=False)
    source=reference['model_state']
    baseline=[json.loads(x) for x in (SOURCE/'ss/history.jsonl').read_text().splitlines()][:500]
    summaries={}
    for name in ('baseline500',*ARMS):
        if name=='baseline500':
            history=baseline;path=SOURCE/'ss/checkpoints/step_000500.pt'
        else:
            history=[json.loads(x) for x in (OUT/name/'history.jsonl').read_text().splitlines()]
            path=OUT/name/'checkpoints/step_000500.pt'
            if json.loads((OUT/name/'complete.json').read_text())['step']!=500:raise ValueError('incomplete pilot')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if [r['step'] for r in history]!=list(range(1,501)) or cp['step']!=500:raise ValueError('step coverage differs')
        if history[-1]['object_group_exposures']!=1000:raise ValueError('exposure budget differs')
        if [(r['epoch'],r['offset']) for r in history] != [(r['epoch'],r['offset']) for r in baseline]:
            raise ValueError('sample cursors differ')
        all_finite=all(torch.isfinite(v).all() for v in cp['model_state'].values())
        if not all_finite:raise ValueError('nonfinite checkpoint')
        geometry={k[len('target_lari.'):]:v for k,v in cp['model_state'].items() if k.startswith('target_lari.')}
        frozen_equal=set(geometry)==set(source) and all(torch.equal(geometry[k],source[k]) for k in source)
        if cp['args']['freeze_geometry'] and not frozen_equal:raise ValueError('frozen geometry changed')
        fields=('depth_mae','layer_loss','cross_view_loss','flow_loss','loss','gradient_norm','target_support_mean','diffusion_time')
        windows={}
        for key,rr in [('first100',history[:100]),('last100',history[-100:])]:
            windows[key]={f:float(np.mean([r[f] for r in rr])) for f in fields if all(f in r for r in rr)}
        summaries[name]={'checkpoint':binding(path),'steps':500,'exposures':1000,'parameters_finite':True,
                         'geometry_bit_exact_to_lari_raw':frozen_equal,'windows':windows,
                         'elapsed_seconds':history[-1]['elapsed_seconds'],'args':cp['args']}
    atomic_json(OUT/'training_aggregate.json',{'arms':summaries,'matched_cursors':True,
        'scope':'500-step pilot, one training seed; flow losses across different time distributions are not comparable'})
    print(json.dumps({k:v['windows']['last100'] for k,v in summaries.items()}))


if __name__=='__main__':main()
