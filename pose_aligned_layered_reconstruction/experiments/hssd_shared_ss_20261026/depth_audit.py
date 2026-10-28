"""Evaluation-only RGB depth check. Never imported by model/data/train."""
import numpy as np
from .common import *


def main():
    p=plan()['plan'];rows=[]
    for group,frames in p['depth_policy']['groups'].items():
        receipt=read(OUT/'depth'/f'{group}.json')
        with np.load(verify(receipt['npz'])) as d:
            pred=d['depth_z_world'].copy();rank=d['rank_conf'].copy()
        for i,frame in enumerate(frames):
            gt_binding=next(b for b in p['evaluation_gt'] if Path(b['path']).stem==f'{frame:03d}')
            with np.load(verify(gt_binding)) as a:gt=a['depth_z_world']
            mask=(gt>0)&np.isfinite(gt)&np.isfinite(pred[i])&(pred[i]>0)
            values={}
            for name,m in [('all_valid_gt',mask),('rank_ge_0p3',mask&(rank[i]>=.3))]:
                x,y=pred[i][m].astype(np.float64),gt[m].astype(np.float64)
                if not len(x):raise ValueError('Depth audit has no valid pixels')
                values[name]={'pixels':len(x),'abs_relative':float(np.mean(np.abs(x-y)/y)),
                    'rmse_world':float(np.sqrt(np.mean((x-y)**2))),
                    'median_pred_over_gt':float(np.median(x/y)),
                    'delta_1p25':float(np.mean(np.maximum(x/y,y/x)<1.25))}
            rows.append({'group':group,'frame':frame,'depth':binding(OUT/'depth'/f'{group}.json'),
                'gt_for_evaluation_only':gt_binding,'metrics':values})
    write(OUT/'depth_accuracy_eval_only.json',{'time':now(),'rows':rows,
        'scope':'Post-cache evaluation only, no per-frame or per-object GT fit, no thresholds tuned from these values; not fed to SS model or checkpoint selection',
        'scene_mesh_or_target_support_used_to_correct_prediction':False})
    event('depth_accuracy_audit_complete',frames=len(rows),used_for_training=False)


if __name__=='__main__':main()
