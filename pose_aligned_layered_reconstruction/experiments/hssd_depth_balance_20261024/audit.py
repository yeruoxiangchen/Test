"""FP32 LaRI measurements for both newly trained predictors on the original16."""
import json
import torch
from ...v2.dataset import SceneDataset, select_views
from ...v2.layer_predictor import RayLayerPredictor, GeometryConfig
from ...v2.train import geometry_forward
from ..hssd_lari_causal_20261023.audit import audit_sample
from .common import ARMS, OUT, LARI_STEPS, FORMAT, freeze, read, binding, verify, write, locked


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    protocol=freeze()
    dataset=SceneDataset(verify(protocol['manifest']), require_labels=True)
    summary={'scope':'same16 training objects; FP32 final raw weights, no fitted alignment', 'arms':{}}
    for arm in ARMS:
        path=OUT/'train'/f'lari_{arm}'/'last.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['format']!=FORMAT or cp['step']!=LARI_STEPS or cp['contract']['protocol']!=protocol or cp['contract']['arm']!=arm:
            raise ValueError('Require complete matching LaRI')
        root=OUT/'audit'/arm
        locked(root/'identity.json',{'protocol':protocol,'checkpoint':binding(path)})
        model=RayLayerPredictor(GeometryConfig(**protocol['geometry_config'])).eval().to('cuda:0')
        model.load_state_dict(cp['state'],strict=True)
        rows=[]
        for index in range(16):
            dest=root/f'{index:03d}.json'
            sample=select_views(dataset[index],torch.arange(8))
            if dest.exists():
                row=read(dest)
                verify(row['prediction'])
            else:
                pred,_=geometry_forward(model,sample,'cuda:0')
                pred=pred.to('cpu')
                saved=root/f'{index:03d}.pt'
                torch.save({k:getattr(pred,k) for k in ['depths','stop_logits','target_logits','confidence_logits']},saved)
                row={'index':index,'uid':sample['uid'],'prediction':binding(saved),**audit_sample(sample,pred)}
                write(dest,row)
            if row['uid']!=sample['uid']:raise ValueError('Wrong audit object')
            rows.append(row)
            print(json.dumps({'stage':'lari_audit','arm':arm,'index':index}),flush=True)
        def mean(values):
            values=[v for v in values if v is not None]
            return sum(values)/len(values) if values else None
        summary['arms'][arm]={'parts':{part:{key:mean([r['parts'][part][key] for r in rows])
            for key in rows[0]['parts'][part]} for part in rows[0]['parts']},
            'records':rows,'stop_accuracy':mean([r['stop']['accuracy'] for r in rows])}
        del model,cp
        torch.cuda.empty_cache()
    summary['target_mae_improvement']=summary['arms']['uniform']['parts']['target']['mae']-summary['arms']['balanced']['parts']['target']['mae']
    summary['hidden_mae_improvement']=summary['arms']['uniform']['parts']['hidden_behind_non_target']['mae']-summary['arms']['balanced']['parts']['hidden_behind_non_target']['mae']
    summary['automatic_expansion']=False
    write(OUT/'audit_summary.json',summary)


if __name__=='__main__':main()
