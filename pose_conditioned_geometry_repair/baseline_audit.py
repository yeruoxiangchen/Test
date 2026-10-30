"""Re-evaluate the physical damaged input without the teacher support shortcut."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from . import common as c
from .data import RepairDataset,intervention
from .evaluate import canonical_gt,shape_mesh,mesh_metrics

V1=c.ROOT/'outputs/round1_20261029'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output).resolve()
    if not out.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Audit output must be in project storage')
    from trellis2 import models
    from trellis2.modules import sparse as sp
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    decoder=models.from_pretrained(c.MODEL_PATHS['shape_decoder']).eval().cuda()
    decoder.set_resolution(512)
    decoder.low_vram=True
    ds=RepairDataset(c.DATA/'overfit.json','shape')
    plan={r['object_uid']:r for r in c.plan()['plan']['samples']}
    records=[]
    for index in range(len(ds)):
        sample=ds[index]
        folder=out/'objects'/sample['uid']
        result=folder/'metrics.json'
        if result.exists():
            records.append(c.read(result))
            continue
        folder.mkdir(parents=True,exist_ok=True)
        pair=intervention(sample,'shape',0,'cuda')
        present=pair['present']
        if not present.any():
            raise ValueError('Synthetic damaged input has empty support')
        bad=sp.SparseTensor(pair['bad'].feats[present],pair['bad'].coords[present])
        mesh=shape_mesh(bad,decoder,folder/'damaged_input.obj')
        gt=canonical_gt(plan[sample['uid']])
        metric=mesh_metrics(mesh,gt)
        v1=c.read(V1/'eval_overfit_repair_shape_step_002000_s42/objects'/sample['uid']/'shape_metrics_0.json')
        rec={'uid':sample['uid'],'time':c.now(),'damaged_input':metric,
             'v1_metrics':v1['metrics'],'actual_input_tokens':int(present.sum()),
             'teacher_support_tokens':len(present),
             'reason':'Missing reference cells are absent, rather than zero-feature cells with GT support'}
        c.write(result,rec)
        records.append(rec)
        c.write(out/'progress.json',{'time':c.now(),'records':len(records),'expected':32})
        print(json.dumps({'uid':sample['uid'],'damaged_input_cd':metric['cd_l1'],
            'v1_rollout_cd':v1['metrics']['candidate_predicted_ss']['cd_l1']}),flush=True)
    means={'damaged_input':float(np.mean([r['damaged_input']['cd_l1'] for r in records]))}
    for arm in records[0]['v1_metrics']:
        means[arm]=float(np.mean([r['v1_metrics'][arm]['cd_l1'] for r in records]))
    gain=1-means['candidate_predicted_ss']/means['damaged_input']
    c.write(out/'summary.json',{'time':c.now(),'complete':len(records)==32,'records':len(records),
        'mean_cd':means,'v1_predicted_ss_gain_vs_actual_damaged_input':gain,
        'does_not_change_teacher_support_failure':True,
        'note':'V1 is not retroactively marked passed. Its teacher-support mesh and overfit gate failed.'})


if __name__=='__main__':
    main()
