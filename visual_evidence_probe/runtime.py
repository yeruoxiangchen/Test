import gc
import time

import numpy as np
import torch
from torch import nn

from .common import binding, verify, folder, write, emit


def setup():
    from pose_aligned_layered_reconstruction._bootstrap import bootstrap_external_paths
    bootstrap_external_paths()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda:0')


def load_ss(device):
    from peft import LoraConfig, get_peft_model
    from pose_aligned_reconstruction.current_30k import SS30K_CHECKPOINT, PRETRAINED
    from pose_aligned_reconstruction.native_ss_genrecon import NativeSSGenreconFlow, load_trainable_state_dict
    from pose_aligned_layered_reconstruction.pretrained import load_trellis_model
    from trellis.pipelines import samplers
    SS30K_CHECKPOINT.validate(full_hash=True)
    cp = torch.load(SS30K_CHECKPOINT.path, map_location='cpu', mmap=True, weights_only=False)
    if cp['step'] != 30000:
        raise ValueError('Wrong SS step')
    base, b = load_trellis_model(PRETRAINED, 'ss_flow_img_dit_L_16l8_fp16')
    base.eval().requires_grad_(False)
    base.use_checkpoint = False
    for block in base.blocks:
        block.use_checkpoint = False
    flow = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
        target_modules=['to_q', 'to_kv', 'to_out', 'to_qkv']))
    model = NativeSSGenreconFlow(flow, condition_channels=1024)
    load_trainable_state_dict(model, cp['ema_trainable_state'])
    model.eval().requires_grad_(False).to(device)
    decoder, db = load_trellis_model(PRETRAINED, 'ss_dec_conv3d_16l8_fp16')
    decoder.eval().requires_grad_(False).to(device)
    return {'model': model, 'decoder': decoder,
            'sampler': samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1.e-5),
            'binding': {'checkpoint': binding(SS30K_CHECKPOINT.path), 'weights': 'ema', 'step': 30000,
                        'base': b, 'decoder': db}}


class CachedSS(nn.Module):
    """Cache image-only projection; preserve the parent's adapted equations."""
    def __init__(self, model, positive, sample):
        super().__init__()
        from pose_aligned_reconstruction.native_ss_genrecon import project_frustum_dino
        self.model, self.positive = model, positive
        projected, valid, _ = project_frustum_dino(sample, device=positive.device)
        self.condition, _ = model.aggregator(projected, valid)
        self.null = torch.zeros_like(self.condition)

    def forward(self, x, t, condition):
        positive = condition is self.positive or (torch.is_tensor(condition)
            and condition.shape == self.positive.shape and condition.data_ptr() == self.positive.data_ptr())
        return self.model._adapted_core_forward(x, t, condition,
            self.condition if positive else self.null)


@torch.no_grad()
def sample_ss(runtime, sample, index, seed, output):
    from pose_aligned_layered_reconstruction.evaluate_ss_occupancy import occupancy_metrics
    device = next(runtime['model'].parameters()).device
    positive = sample['stock_condition'].to(device)
    generator = torch.Generator(device=device).manual_seed(seed * 1000003 + index * 1009)
    initial = torch.randn((1, 8, 16, 16, 16), generator=generator, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        wrapper = CachedSS(runtime['model'], positive, sample)
        z = runtime['sampler'].sample(wrapper, initial, cond=positive, neg_cond=torch.zeros_like(positive),
            steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
        logits = runtime['decoder'](z.to(next(runtime['decoder'].parameters()).dtype))
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite SS')
    coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)
    if len(coords) == 0:
        raise ValueError('Empty SS (retain failure, do not skip)')
    np.savez_compressed(output, coords=coords)
    return coords


def load_slat(device):
    from pose_aligned_layered_reconstruction.frozen_slat import load_frozen_slat30k
    from pose_aligned_layered_reconstruction.experiments.round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
    patch = install_memory_efficient_mesh()
    runtime = load_frozen_slat30k(device)
    runtime['binding']['memory_efficient_mesh'] = patch
    return runtime


@torch.no_grad()
def mesh_from_support(runtime, sample, coords, index, seed, path):
    from pose_aligned_layered_reconstruction.frozen_slat import master_noise, sparse_noise, sample_mesh_latent
    initial = sparse_noise(coords, master_noise(seed, index, runtime['mean'].device))
    z, stats = sample_mesh_latent(runtime, sample, initial)
    # The original decoder must run outside autocast, with the parent's denormalized latent.
    mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError('Invalid mesh (retain failure, do not skip)')
    mesh.export(path, file_type='obj')
    return mesh, stats


@torch.no_grad()
def parity(runtime, sample, index):
    from pose_aligned_reconstruction.native_ss_genrecon import NativeSSCalibratedCFGFlow
    model = runtime['model']
    positive = sample['stock_condition'].to(next(model.parameters()).device)
    original = NativeSSCalibratedCFGFlow(model, positive, sample)
    noise = torch.randn((1, 8, 16, 16, 16), device=positive.device,
                        generator=torch.Generator(device=positive.device).manual_seed(90210))
    t = torch.tensor([500.], device=positive.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cached = CachedSS(model, positive, sample)
        a, b = original(noise, t, positive), cached(noise, t, positive)
        negative = torch.zeros_like(positive)
        an, bn = original(noise, t, negative), cached(noise, t, negative)
    result = {'positive_max_abs': float((a - b).abs().max()),
              'negative_max_abs': float((an - bn).abs().max()),
              'exact': torch.equal(a, b) and torch.equal(an, bn)}
    if not result['exact']:
        raise ValueError(f'Cached SS changed parent equations: {result}')
    return result
