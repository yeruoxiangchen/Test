"""Two native SS replays and bounded intervention forwards; no training/mesh."""
import json
import numpy as np
import torch
from ...v2.dataset import SceneDataset,select_views
from ...pretrained import load_trellis_model
from ..hssd_cell_query_20261023.model import ResidualCFG
from .common import OUT,CELL,PILOT,ADAPTERS,VARIANTS,read,verify,binding,write,freeze
from .interventions import prepare_variants
from .ss_probe import load_model


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    protocol=freeze()
    device=torch.device('cuda:0')
    source=select_views(SceneDataset(PILOT/'occluded16.json',require_labels=True,require_ss_target=True)[0],torch.arange(8))
    sampler,model,saved=load_model('predicted_point',device)
    decoder,decoder_binding=load_trellis_model(saved['pretrained'],'ss_dec_conv3d_16l8_fp16')
    decoder=decoder.to(device).eval().requires_grad_(False)
    context=source['stock_condition'].to(device)
    generator=torch.Generator(device=device).manual_seed(42*1000003)
    noise=torch.randn((1,8,16,16,16),device=device,generator=generator)
    target=source['target'].to(device).float()[None]
    xt,velocity=sampler._get_model_gt(target,.5,noise)
    result={}
    checkpoints={}
    for adapter in ADAPTERS:
        path=CELL/'train'/f'occluded_{adapter}'/'last.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model.residual.load_state_dict(cp['residual'],strict=True)
        model.arm=adapter
        versions={n:p._version for n,p in model.named_parameters()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prepared,native,parity=prepare_variants(model,source,device)
            t=target.new_tensor([500.])
            baseline=model(xt,t,context,native)
            equivalent=model(xt,t,context,prepared[parity['native_variant']])
            assert torch.equal(baseline,equivalent),'Native forward changed'
            mse={}
            for variant in VARIANTS:
                out=model(xt,t,context,prepared[variant])
                assert torch.isfinite(out).all()
                mse[variant]=float((out.float()-velocity.float()).square().mean())
            wrapper=ResidualCFG(model,context,prepared[parity['native_variant']])
            z=sampler.sample(wrapper,noise,cond=context,neg_cond=torch.zeros_like(context),steps=25,
                cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
            logits=decoder(z.to(next(decoder.parameters()).dtype))
            assert torch.isfinite(logits).all()
        coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
        ref=read(CELL/'evaluations'/f'occluded_{adapter}'/'000'/'42_ss.json')
        with np.load(verify(ref['coords'])) as data:
            assert np.array_equal(coords,data['coords']),'Native SS replay differs'
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        result[adapter]={**parity,'native_forward_exact':True,'native_ss_coords_exact':True,
            'active_voxels':len(coords),'all_frozen_unchanged':True,'single_time_velocity_mse':mse}
        checkpoints[adapter]=binding(path)
        print(json.dumps({'adapter':adapter,**result[adapter]}),flush=True)
    write(OUT/'smoke.json',{'passed':True,'protocol':protocol,'checkpoints':checkpoints,
        'decoder':decoder_binding,'results':result,'scope':'2 native SS trajectories, 16 intervention forwards; no training or mesh quality claim'})
    print('CAUSAL_SMOKE_PASSED',flush=True)


if __name__=='__main__':main()
