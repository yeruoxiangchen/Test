"""Bounded checks using the actual training entry and one real SS trajectory."""
import argparse
import gc
import json
import subprocess
from pathlib import Path
import torch
from ...v2.dataset import SceneDataset, select_views
from ...pretrained import load_trellis_model
from ..hssd_cell_query_20261023.train import fixed_noise
from .common import ARMS, OUT, MODULE, PYTHON, freeze, sources, policy, tensor_hash, binding, read, write, now
from .model import build, observations, ResidualCFG


def archive_changed_preflight():
    path=OUT/'protocol.json'
    if not path.exists() or read(path)['sources']==sources():return
    if any((OUT/name).exists() for name in ['train','evaluations','mesh','audit']):
        raise ValueError('Code changed after formal outputs: require a new experiment version')
    archive=OUT/'superseded_preflight'/now().replace(':','').replace('+','_')
    archive.mkdir(parents=True)
    for name in ['protocol.json','smoke.json','preflight']:
        source=OUT/name
        if source.exists():source.rename(archive/name)
    print('Archived incomplete engineering preflight: '+str(archive),flush=True)


def invoke(stage,*args):
    log=OUT/'preflight'/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[PYTHON,'-u','-m',MODULE+'.train',*args]
    with log.open('a') as stream:
        stream.write(json.dumps({'time':now(),'command':cmd})+'\n');stream.flush()
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Preflight entry passed: '+stage,flush=True)


@torch.no_grad()
def forward_check(arm):
    torch.set_num_threads(2)
    lari=OUT/'preflight'/f'lari_{arm}'/'last.pt'
    contract=policy('ss',arm,steps=2,lari_path=lari)
    sampler,model,saved,bindings=build(contract,torch.device('cuda:0'))
    dataset=SceneDataset(contract['manifest']['path'],require_labels=True,require_ss_target=True)
    sample=select_views(dataset[0],torch.arange(8));obs=observations(sample)
    cp=torch.load(OUT/'preflight'/f'ss_{arm}'/'last.pt',map_location='cpu',weights_only=False)
    assert cp['contract']==contract and cp['bindings']==bindings and cp['step']==2
    before=tensor_hash(model.base.target_lari.state_dict())
    target=sample['target'].to('cuda:0').float()[None];positive=sample['stock_condition'].to('cuda:0')
    t,noise=fixed_noise(target,contract['seed'],0);xt,_=sampler._get_model_gt(target,t,noise)
    tm=target.new_tensor([t*1000])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prepared=model.prepare(obs,'cuda:0')
        # Labels may exist in the source sample, but are excluded from this branch.
        with_labels=model.prepare(sample,'cuda:0')
        assert all(torch.equal(v,with_labels[k]) for k,v in prepared.items())
        parent,layer,_=model.base.build_conditions(obs,device='cuda:0')
        assert torch.equal(parent,prepared['appearance']) and torch.equal(layer,prepared['appearance_layer'])
        baseline=model.base._core_forward(xt,tm,positive,parent,layer)
        zero=model(xt,tm,positive,prepared)
        assert torch.equal(zero,baseline)
        model.residual.load_state_dict(cp['state'],strict=True)
        output=model(xt,tm,positive,prepared)
        assert torch.isfinite(output).all()
        assert tensor_hash(model.residual.state_dict())==tensor_hash(cp['state'])
        assert tensor_hash(model.base.target_lari.state_dict())==before
        wrapper=ResidualCFG(model,positive,prepared)
        negative=wrapper(xt,tm,torch.zeros_like(positive))
        expected=model.base.adapted_prediction(xt,tm,torch.zeros_like(positive),None)[0]
        assert torch.equal(negative,expected)
        result={'zero_init_exact':True,'observations_only_exact':True,'negative_cfg_exact':True,
                'restore_exact':True,'lari_state_sha':before,'finite_forward':True}
        if arm=='balanced':
            latent=sampler.sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
            decoder,_=load_trellis_model(saved['pretrained'],'ss_dec_conv3d_16l8_fp16')
            decoder=decoder.to('cuda:0').eval()
            logits=decoder(latent.to(next(decoder.parameters()).dtype))
            assert torch.isfinite(latent).all() and torch.isfinite(logits).all() and (logits>0).any()
            result.update(ss_steps=25,active_voxels=int((logits>0).sum()),positive_calls=wrapper.positive_calls,
                          negative_calls=wrapper.negative_calls)
    write(OUT/'preflight'/f'forward_{arm}.json',result)
    print(json.dumps({'arm':arm,**result}),flush=True)


def main():
    archive_changed_preflight();protocol=freeze()
    evidence=[];pairs={}
    for kind in ['lari','ss']:
        pairs[kind]=[]
        for arm in ARMS:
            root=OUT/'preflight'/f'{kind}_{arm}'
            args=[kind,'--arm',arm,'--steps','2','--output',str(root)]
            if kind=='ss':args+=['--lari-path',str(OUT/'preflight'/f'lari_{arm}'/'last.pt')]
            if not (root/'last.pt').exists():invoke(f'{kind}_{arm}_first',*args,'--limit-updates','1')
            invoke(f'{kind}_{arm}_resume',*args)
            cp=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            run=read(root/'run.json');history=[json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
            assert cp['step']==2 and [r['step'] for r in history]==[1,2]
            assert all(r['gradient_norm']>0 for r in history)
            original=[binding(root/name) for name in ['last.pt','history.jsonl','run.json']]
            invoke(f'{kind}_{arm}_complete_reentry',*args)
            assert original==[binding(row['path']) for row in original]
            pairs[kind].append({'initial_sha':run['initial_state_sha'],'parameter_count':run['trainable_parameter_count'],
                'order':[(r['uid'],r['learning_rate'],r.get('diffusion_time')) for r in history]})
            evidence+=original+[binding(root/'complete.json')]
            del cp;gc.collect()
        assert pairs[kind][0]==pairs[kind][1]
    for arm in ARMS:
        log=OUT/'preflight'/f'forward_{arm}.log'
        with log.open('a') as stream:
            subprocess.run([PYTHON,'-u','-m',MODULE+'.smoke','--forward',arm],stdout=stream,stderr=subprocess.STDOUT,check=True)
        evidence.append(binding(OUT/'preflight'/f'forward_{arm}.json'))
    write(OUT/'smoke.json',{'passed':True,'time':now(),'protocol':protocol,'evidence':evidence,'paired_training':pairs,
        'scope':'two updates per arm and stage, resumed real entry, exact completed reentry; one SS25; no formal training or mesh'})
    print('DEPTH_BALANCE_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--forward',choices=ARMS);a=p.parse_args()
    forward_check(a.forward) if a.forward else main()
