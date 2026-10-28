"""Remove geometric evidence but keep trained positive-branch projection biases."""
import argparse
import json
import os
import time
import numpy as np
import torch
from .diagnostic_suite import (OUT, SOURCE, OLD, KEYS, binding, verify, bind_code,
                               atomic_json, canonical_json_sha256, SceneDataset, select_views)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, choices=range(8), default=1)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2)
    from ...v2.ss import build_ss
    from ...v2.checkpoints import read_checkpoint, load_state
    from ...v2.layer_predictor import GeometryConfig
    from ...native_ss import LayeredSSCFGFlow
    from ...pretrained import load_trellis_model
    from ...evaluate_ss_occupancy import occupancy_metrics
    root=OUT/'dev64/bias_only';device=torch.device('cuda:0')
    manifest=OLD/'data_v2/dev64.json';checkpoint=SOURCE/'ss/checkpoints/step_002000.pt'
    identity={'manifest':binding(manifest),'checkpoint':binding(checkpoint),'code':bind_code(),
              'probe_code':binding(__file__),'mode':'bias_only','seeds':[42,43],'steps':25,
              'intervention':'parent appearance aggregation, zero layer embedding, trained projection biases retained'}
    digest=canonical_json_sha256(identity)
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())['identity_hash']!=digest:
        raise ValueError('probe identity changed')
    atomic_json(root/'identity.json',{'identity':identity,'identity_hash':digest})
    saved=read_checkpoint(checkpoint);cfg=GeometryConfig(**saved['geometry_config'])
    sampler,model,bindings=build_ss(cfg,device,saved['args']['parent_ss30k'],saved['args']['pretrained'],False)
    load_state(model,saved,cfg,stage='ss',bindings=bindings);model.eval()
    decoder,_=load_trellis_model(saved['args']['pretrained'],'ss_dec_conv3d_16l8_fp16')
    decoder.to(device).eval().requires_grad_(False);del saved
    dataset=SceneDataset(manifest,require_labels=False,require_ss_target=True)
    SceneDataset(SOURCE/'train96.json').assert_disjoint(dataset)
    for i in range(len(dataset)):
        original=dataset[i]
        sample=select_views({k:original[k] for k in KEYS if k in original},torch.arange(8))
        positive=sample['stock_condition'].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            appearance,_,_=model.build_conditions(sample,device=device,use_layer_branch=False)
            full_conditions=model.build_conditions(sample,device=device,use_layer_branch=True)
            signals=[];biases=[]
            for projection in model.layer_block_condition.projections:
                signal=torch.nn.functional.linear(full_conditions[1],projection.weight,None)
                signals.append(float(signal.float().square().mean()))
                biases.append(float(projection.bias.float().square().mean()))
        atomic_json(root/'condition_stats'/f'{i:03d}.json',{
            'identity_hash':digest,'index':i,'object_uid':original['object_uid'],
            'target_support_mean':float(full_conditions[2]['target_support_mean']),
            'geometry_feature_residual_rms':float(np.sqrt(np.mean(signals))),
            'trained_projection_bias_rms':float(np.sqrt(np.mean(biases))),
            'appearance_gate_delta_rms':float((full_conditions[0]-appearance).float().square().mean().sqrt())})
        layer=appearance.new_zeros((*appearance.shape[:-1],model.layer_encoder.output_channels))
        for seed in (42,43):
            path=root/'records'/f'{i:03d}_{seed}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['identity_hash']!=digest:raise ValueError('probe record changed')
                verify(old['coordinates']);continue
            start=time.time()
            g=torch.Generator(device=device).manual_seed(seed*1000003+int(original['object_uid'][:8],16))
            noise=torch.randn((1,8,16,16,16),device=device,generator=g)
            if i==0 and seed==42:
                # A fresh full-model reproduction anchors this intervention to the existing paired run.
                reference=LayeredSSCFGFlow(model,positive,sample);reference.cached_conditions=full_conditions[:2]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    rz=sampler.sample(reference,noise.clone(),cond=positive,neg_cond=torch.zeros_like(positive),steps=25,
                        cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
                    rl=decoder(rz.to(next(decoder.parameters()).dtype))
                rc=torch.argwhere(rl>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
                record=json.loads((OUT/'dev64/raw/records/000_42.json').read_text())
                with np.load(verify(record['coordinates'])) as data:
                    exact=np.array_equal(rc,data['coords'])
                if not exact:raise ValueError('full-model reproduction differs before bias intervention')
                atomic_json(root/'full_model_reproduction.json',{'coordinates_bit_exact':True,'index':0,'seed':42})
                del reference,rz,rl
            flow=LayeredSSCFGFlow(model,positive,sample);flow.cached_conditions=(appearance,layer)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=sampler.sample(flow,noise,cond=positive,neg_cond=torch.zeros_like(positive),steps=25,
                    cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
                logits=decoder(z.to(next(decoder.parameters()).dtype))
            if not torch.isfinite(z).all() or not torch.isfinite(logits).all():raise FloatingPointError('invalid probe')
            coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
            target=root/'coordinates'/f'{i:03d}_{seed}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,coords=coords)
            record={'identity_hash':digest,'index':i,'seed':seed,'mode':'bias_only',
                    'uid':original['uid'],'object_uid':original['object_uid'],'domain':original['data_domain'],
                    **occupancy_metrics(coords,original['target_coords'].numpy()),'coordinates':binding(target),
                    'elapsed_seconds':time.time()-start}
            atomic_json(path,record)
            print(json.dumps({k:record[k] for k in ('index','seed','iou','elapsed_seconds')}),flush=True)
    atomic_json(root/'complete.json',{'records':128,'identity_hash':digest})


if __name__=='__main__':main()
