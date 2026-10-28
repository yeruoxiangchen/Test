"""Bounded real-GPU checks: exact initial fallback, six updates, CFG/SS decode."""
import json
import torch
import torch.nn.functional as F

from ...v2.dataset import SceneDataset, select_views
from ...pretrained import load_trellis_model
from .common import ARMS, OUT, SEED, policy, sources, verify_old_sources, tensor_hash, write, now
from .model import build, NestedSS, ResidualCFG, observations
from .train import fixed_noise


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    contract = policy('occluded', 'appearance_control')
    original = verify_old_sources()
    dataset = SceneDataset(contract['manifest']['path'], require_labels=True, require_ss_target=True)
    sample = select_views(dataset[0], torch.arange(8))
    sampler, model, saved, _ = build(contract, device, checkpointing=True)
    base = model.base
    versions = {n: p._version for n, p in base.named_parameters()}
    target = sample['target'].to(device).float()[None]
    context = sample['stock_condition'].to(device)
    t, noise = fixed_noise(target, SEED, 0)
    xt, velocity = sampler._get_model_gt(target, t, noise)
    tm = target.new_tensor([t * 1000])
    results = {}
    for arm in ARMS:
        model = NestedSS(base, arm).to(device).train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prepared = model.prepare(sample if arm == 'oracle' else observations(sample), device)
            parent, layer, _ = base.build_conditions(observations(sample), device=device)
            assert torch.equal(parent, prepared['appearance'])
            assert torch.equal(layer, prepared['appearance_layer'])
            with torch.no_grad():
                baseline = base._core_forward(xt, tm, context, parent, layer)
                zero_output = model(xt, tm, context, prepared)
            if not torch.equal(baseline, zero_output):
                raise AssertionError(f'Zero-init changed the appearance base in {arm}')
        init_hash = tensor_hash(model.residual.state_dict())
        optimizer = torch.optim.AdamW(model.residual.parameters(), lr=1.e-4, betas=(.9, .95))
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(xt, tm, context, prepared)
                loss = F.mse_loss(output.float(), velocity.float())
            loss.backward()
            for name, param in model.residual.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise AssertionError('Missing/nonfinite residual gradient: ' + name)
            encoder_grad = sum(float(p.grad.float().abs().sum()) for p in model.residual.encoder.parameters())
            projection_grad = sum(float(p.weight.grad.float().abs().sum()) for p in model.residual.projections)
            if projection_grad <= 0 or (step == 1 and encoder_grad <= 0):
                raise AssertionError('Residual branch failed to receive gradients')
            norm = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps.append({'loss': float(loss.detach()), 'gradient_norm': float(norm),
                          'encoder_gradient_l1': encoder_grad, 'projection_gradient_l1': projection_grad})
        if any(p._version != versions[n] or p.grad is not None for n, p in base.named_parameters()):
            raise AssertionError('Frozen appearance base modified during smoke')
        state = {n: p.detach().cpu().clone() for n, p in model.residual.state_dict().items()}
        restored = NestedSS(base, arm).to(device)
        restored.residual.load_state_dict(state, strict=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            assert torch.equal(model(xt, tm, context, prepared), restored(xt, tm, context, prepared))
        results[arm] = {'initial_sha': init_hash, 'trainable_parameters': sum(p.numel() for p in model.residual.parameters()),
            'zero_init_exact': True, 'restore_exact': True, 'steps': steps, 'frozen_base_unchanged': True}
        print(json.dumps({'arm': arm, **results[arm]}), flush=True)
    if len({r['initial_sha'] for r in results.values()}) != 1 or len({r['trainable_parameters'] for r in results.values()}) != 1:
        raise AssertionError('Unmatched residual initialization/capacity')
    model.eval()
    wrapper = ResidualCFG(model, context, prepared)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Real configured SS trajectory, but no expensive SLat/mesh generation.
        latent = sampler.sample(wrapper, noise, cond=context, neg_cond=torch.zeros_like(context),
            steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
        negative = wrapper(xt, tm, torch.zeros_like(context))
        reference = base.adapted_prediction(xt, tm, torch.zeros_like(context), None)[0]
        assert torch.equal(negative, reference)
    decoder, _ = load_trellis_model(saved['pretrained'], 'ss_dec_conv3d_16l8_fp16')
    decoder = decoder.to(device).eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = decoder(latent.to(next(decoder.parameters()).dtype))
    if not torch.isfinite(logits).all() or not torch.isfinite(latent).all() or wrapper.negative_calls < 1:
        raise AssertionError('Invalid SS trajectory / CFG route')
    write(OUT / 'smoke.json', {'passed': True, 'time': now(), 'sources': sources(), 'original_sources': original,
        'manifest': contract['manifest'], 'base': contract['appearance_base'], 'arms': results,
        'ss_steps': 25, 'ss_logit_shape': list(logits.shape), 'active_voxels': int((logits > 0).sum()),
        'positive_calls': wrapper.positive_calls, 'negative_calls': wrapper.negative_calls,
        'negative_exact': True, 'scope': 'engineering smoke only; six updates, no mesh fit claim'})
    print('RESIDUAL_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    main()
