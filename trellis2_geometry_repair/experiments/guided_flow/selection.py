import argparse
from pathlib import Path
import time
import torch

from .config import OUT,PARAMS,METHOD,DIAGNOSTIC,case,previous,original,read,write,save,digest,now,setup,lock_case,emit
from .data import load,record
from ...runtime import load_model,save_mesh
from ...geometry import transform
from ..local_regeneration.fields import decode_fields,to_mesh,compose
from ..local_regeneration.selection import row_metrics,decision
from ..repair_diagnostic.runner import initial_surface_parity,link

def budget_prefix(rows,input_cost):
    chosen=[];cost=0.
    for row in rows:
        chosen.append(row);cost+=row['sampling_seconds']+row['score_seconds']
        if len(chosen)>=2 and cost>=input_cost:break
    return chosen,dict(input_seconds=input_cost,prior_seconds=cost,prior_candidates=len(chosen),
                      sufficient=cost>=input_cost,ratio=cost/max(input_cost,1.e-9),GT_used=False)

def one(index,seed,mode,context):
    root=case(index,seed);path=root/f'{mode}_complete.json'
    if path.exists():return
    started=time.monotonic();data=load(index,seed,context)
    if mode=='diagnostic' and digest(root/'method_complete.json')!=read(root/'method_sealed.json')['decision_sha256']:raise RuntimeError('method seal changed')
    specs=read(root/f'generation_{mode}.json')['candidates'];decoder=load_model('shape_decoder',binding_root=OUT)
    old=torch.load(original(index,seed)/'original_latent.pt',weights_only=False)
    with torch.no_grad():
        v,f,base_fields=decode_fields(decoder,old['features'].cuda(),old['coords'].cuda())
        initial_surface_parity(v,f,data['native'],data['faces'])
        base_render=data['baseline'];bases={};rows=[];targets={};domains={}
        for arm in (('input',) if mode=='method' else DIAGNOSTIC):
            rec,metadata=record(data,None,arm,index,seed);targets[arm]=rec['trusted'];domains[arm]=metadata
            bases[arm]=row_metrics(v,f,data,data['matrix'],data['roi'],targets[arm],base_render,context)
            bases[arm].update(name='base',kind='direct_native_decode')
        for spec in specs:
            score_arm='input' if mode=='method' else spec['arm'];begin=time.monotonic()
            payload=torch.load(spec['path'],weights_only=False)
            v,f,_=decode_fields(decoder,payload['features'].cuda(),payload['coords'].cuda())
            if not len(f):raise RuntimeError('empty V3 native proposal')
            row=row_metrics(v,f,data,data['matrix'],data['roi'],targets[score_arm],base_render,context)
            row.update(name=spec['name'],arm=spec['arm'],path=spec['path'],kind='direct_native_decode',sampling_seconds=spec['seconds'])
            row['decision']=decision(row,bases[score_arm],int(targets[score_arm]['valid'].sum()),False)
            row['score_seconds']=time.monotonic()-begin;rows.append(row)
            emit(index,seed,'scored',mode=mode,name=row['name'],accepted=row['decision']['accepted'])
            write(root/f'scoring_{mode}_progress.json',dict(time=now(),rows=rows))
        budget=None
        if mode=='method':
            priors=[r for r in rows if r['arm']=='prior'];inputs=[r for r in rows if r['arm']=='input']
            input_cost=sum(r['sampling_seconds']+r['score_seconds'] for r in inputs)
            matched,budget=budget_prefix(priors,input_cost)
            groups=dict(prior=priors[:2],budget_prior=matched,input=inputs)
        else:groups={arm:[r for r in rows if r['arm']==arm] for arm in DIAGNOSTIC}
        selections={};export_cache={};exports=root/'exports';exports.mkdir(exist_ok=True)
        base_mesh=previous(index,seed)/'registered_base.obj';alpha=torch.as_tensor(data['roi']['alpha'],device='cuda:0')
        for arm,choices in groups.items():
            base=bases['input' if mode=='method' else arm];eligible=[r for r in choices if r['decision']['accepted']]
            accepted=min(eligible,key=lambda r:r['decision']['score']) if eligible else base
            candidate=min(choices,key=lambda r:r['decision']['score'])
            ar=root/arm;ar.mkdir(exist_ok=True);items={}
            for endpoint,choice in (('accepted',accepted),('candidate',candidate)):
                if choice['name']=='base':
                    link(ar/f'{endpoint}.obj',base_mesh)
                    items[endpoint]=dict(name='base',mesh=str(base_mesh),kind='direct_native_decode',latent=str(original(index,seed)/'original_latent.pt'))
                    continue
                if choice['name'] not in export_cache:
                    payload=torch.load(choice['path'],weights_only=False)
                    v,f,fields=decode_fields(decoder,payload['features'].cuda(),payload['coords'].cuda())
                    native=exports/f'{choice["name"]}_native.obj';save_mesh(native,transform(v,data['matrix']),f)
                    merged,checks=compose(base_fields,fields,alpha);v,f=to_mesh(merged)
                    composite=exports/f'{choice["name"]}_composite.obj';save_mesh(composite,transform(v,data['matrix']),f)
                    field=exports/f'{choice["name"]}_composite_fields.pt';save(field,{k:v.cpu() for k,v in merged.items()})
                    export_cache[choice['name']]=dict(name=choice['name'],kind='direct_native_decode',mesh=str(native),latent=choice['path'],
                        composite_mesh=str(composite),composite_fields=str(field),composite_not_method_output=True,field_checks=checks)
                items[endpoint]=export_cache[choice['name']];link(ar/f'{endpoint}.obj',Path(items[endpoint]['mesh']))
            selections[arm]=dict(accepted_edit=accepted['name']!='base',candidates=len(choices),**items)
        result=dict(time=now(),index=index,seed=seed,uid=data['uid'],mode=mode,GT_used=mode=='diagnostic',
                    diagnostic_only=mode=='diagnostic',baseline=bases,selections=selections,rows=rows,budget=budget,domains=domains,
                    seconds=time.monotonic()-started,direct_native_selection_only=True)
        write(path,result)
        if mode=='method':write(root/'method_sealed.json',dict(time=now(),decision_sha256=digest(path),GT_used=False,budget_selected_without_GT=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=('method','diagnostic'),required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed):one(a.index,a.seed,a.mode,dr.RasterizeCudaContext(device='cuda:0'))

if __name__=='__main__':main()
