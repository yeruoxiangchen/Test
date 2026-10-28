"""Independent, inference-only port of the released no-VGGT SLat v2 / 30K EMA.

Only the SS support is variable in the paired experiment. No layered predictor,
ground-truth geometry, refinement, or additional SLat training enters this path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ._bootstrap import bootstrap_external_paths
from .dino_context import build_masked_dino_contexts
from .paired_dataset import sha256_file, canonical_json_sha256
from .pretrained import DEFAULT_PRETRAINED, load_trellis_model
from .projection import project_queries, project_masked_dino
from .view_fusion import PoseViewAggregator, EveryBlockConditionProjection


DEFAULT_SLAT = Path('/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt')
SLAT_SHA256 = 'da8d058bbb1a917a5b91cd338d60f7d7ec15d7a5a211c86250ed514d8c0a0371'
FLOW_STATE_SHA256 = '2e11445a9cb5c1ce2937526576189bbdcbcf13854f634127d29bc2cd8b23124a'
DECODER_STATE_SHA256 = '3372a1a95e874f7cad9d512d2bc6eda156637145e03dce9293a1c78cdfc03c03'
SAMPLER_PARAMS = dict(steps=25, cfg_strength=5., cfg_interval=[.5, 1.], rescale_t=3., guidance_rescale=0.)
INPUT_KEYS = ('masked_visual_patch_features', 'intrinsics', 'extrinsics', 'image_size',
              'grid_transform', 'extrinsics_type', 'camera_forward_sign')


def configure_determinism():
    # Native SparseDownsample uses scatter_reduce(mean), whose atomic CUDA
    # default is nondeterministic even with a fixed noise tensor. Request the
    # deterministic implementation for BOTH paired branches and the oracle.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)


def inference_inputs(sample, device):
    """Explicit allowlist: even loading a labelled sample cannot leak GT."""
    inputs = {key: sample[key] for key in INPUT_KEYS}
    contexts = build_masked_dino_contexts(inputs['masked_visual_patch_features'])['slat_condition']
    contexts = {key: [value.to(device) for value in values] for key, values in contexts.items()}
    return inputs, contexts


def canonical_coords(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] not in (3, 4) or value.dtype.kind not in 'iu':
        raise ValueError('sparse coordinates must be integer [N,3/4]')
    if value.shape[1] == 4 and np.any(value[:, 0] != 0):
        raise ValueError('only sparse batch zero is supported')
    xyz = value[:, -3:]
    if xyz.size and (xyz.min() < 0 or xyz.max() >= 64):
        raise ValueError('coordinates outside the 64-cube')
    ordered = np.unique(xyz, axis=0).astype(np.int32)
    if len(ordered) != len(xyz):
        raise ValueError('duplicate SS coordinates')
    return np.concatenate((np.zeros((len(ordered), 1), np.int32), ordered), axis=1)


def master_noise(seed, index, device):
    # Identical to the historical official SLat paired evaluation protocol.
    generator = torch.Generator(device=device).manual_seed(int(seed) * 2000003 + int(index) * 2017 + 7919)
    return torch.randn((64, 64, 64, 8), device=device, generator=generator, dtype=torch.float32)


def noise_features(coords, master):
    xyz = torch.as_tensor(coords[:, -3:], device=master.device, dtype=torch.long)
    return master[xyz[:, 0], xyz[:, 1], xyz[:, 2]].clone()


def sparse_noise(coords, master):
    bootstrap_external_paths()
    from trellis.modules import sparse as sp
    coords = canonical_coords(coords)
    return sp.SparseTensor(feats=noise_features(coords, master),
                           coords=torch.as_tensor(coords, device=master.device, dtype=torch.int32))


def module_state_sha256(module):
    """Same tensor-level identity as the archived base-flow freeze contract."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(json.dumps(list(value.shape)).encode('ascii'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def project_active_dino(sample, coords, device):
    if coords.ndim != 2 or coords.shape[1] != 4 or bool((coords[:, 0] != 0).any()):
        raise ValueError('SLat stem coordinates must be batch-zero [N,4]')
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 32)).any()):
        raise ValueError('SLat stem coordinates outside active 32-cube')
    points = (coords[:, 1:].to(device=device, dtype=torch.float32) + .5) / 32. - .5
    transform = str(sample['grid_transform'])
    if transform == 'pixal3d_rotation':
        rotation = points.new_tensor(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
        points = points @ rotation.T
    elif transform != 'identity':
        raise ValueError('unsupported SLat grid transform')
    patches = int(sample['masked_visual_patch_features'].shape[1])
    side = int(round(patches ** .5))
    if side * side != patches:
        raise ValueError('DINO patch grid must be square')
    height, width = map(int, sample['image_size'])
    geometry = project_queries(points, intrinsics=sample['intrinsics'].to(device).float(),
        extrinsics=sample['extrinsics'].to(device).float(), extrinsics_type=sample['extrinsics_type'],
        camera_forward_sign=sample['camera_forward_sign'], image_height=height,
        image_width=width, patch_grid_side=side)
    # Keep the released appearance projection convention. In particular, the
    # newer layer-only camera Y flip must NOT be applied to this frozen model.
    return project_masked_dino(sample, geometry, device=device), geometry['valid']


class FrozenSLat30K(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        core = self.flow_core
        if (core.resolution, core.in_channels, core.out_channels, core.patch_size) != (64, 8, 8, 2):
            raise ValueError('unexpected SLat Flow schema')
        self.aggregator = PoseViewAggregator(1024)
        self.block_condition = EveryBlockConditionProjection(1024, int(core.model_channels), len(core.blocks))

    @property
    def flow_core(self):
        return self.flow.base_model.model

    def forward(self, x, t, condition, sample, cache):
        core = self.flow_core
        h = core.input_layer(x).type(core.dtype)
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        t_embedding = t_embedding.type(core.dtype)
        condition = [v.type(core.dtype) for v in condition] if isinstance(condition, list) else condition.type(core.dtype)
        skips = []
        for block in core.input_blocks:
            h = block(h, t_embedding)
            skips.append(h.feats)
        if core.pe_mode == 'ape':
            h = h + core.pos_embedder(h.coords[:, 1:]).type(core.dtype)
        if sample is not None:
            if not cache:
                projected, valid = project_active_dino(sample, h.coords, x.device)
                aggregated, _, _ = self.aggregator(projected, valid)
                cache.update(coords=h.coords.clone(), condition=aggregated[0])
            if not torch.equal(cache['coords'], h.coords):
                raise RuntimeError('active stem coordinate order changed during SLat sampling')
            condition_3d = cache['condition']
        else:
            # Both CFG branches retain trained LoRA and projection biases.
            condition_3d = x.feats.new_zeros((len(h.feats), 1024), dtype=torch.float32)
        for i, block in enumerate(core.blocks):
            residual = self.block_condition(i, condition_3d)
            h = block(h.replace(h.feats + residual.to(h.dtype)), t_embedding, condition)
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat((h.feats, skip), dim=1)), t_embedding)
            else:
                h = block(h, t_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return core.out_layer(h.type(x.dtype))


class FrozenSLatCFG(nn.Module):
    def __init__(self, model, positive, sample):
        super().__init__()
        self.model, self.positive, self.sample = model, positive, sample
        self.cache = {}
        self.positive_calls = self.negative_calls = 0

    def forward(self, x, t, condition):
        positive = condition is self.positive or (
            isinstance(condition, list) and len(condition) == len(self.positive)
            and all(a is b for a, b in zip(condition, self.positive)))
        self.positive_calls += int(positive)
        self.negative_calls += int(not positive)
        return self.model(x, t, condition, self.sample if positive else None, self.cache)


def load_frozen_slat30k(device, checkpoint=DEFAULT_SLAT, evaluation_uids=()):
    configure_determinism()
    bootstrap_external_paths()
    from peft import LoraConfig, get_peft_model
    from trellis.pipelines import samplers
    checkpoint = Path(checkpoint).resolve(strict=True)
    if sha256_file(checkpoint) != SLAT_SHA256:
        raise RuntimeError('checkpoint is not the frozen paper SLat30K')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['step'] != 30000 or saved['format'] != 'pose_point_depth_mv.native_slat_genrecon_no_vggt.v1':
        raise RuntimeError('SLat version/step differs')
    training = set(saved['data_identity']['object_uids'])
    if set(evaluation_uids) & training:
        raise RuntimeError('SLat30K training/evaluation UID overlap')
    base, flow_binding = load_trellis_model(DEFAULT_PRETRAINED, 'slat_flow_img_dit_L_64l8p2_fp16')
    decoder, decoder_binding = load_trellis_model(DEFAULT_PRETRAINED, 'slat_dec_mesh_swin8_B_64l8m256c_fp16')
    if module_state_sha256(base) != FLOW_STATE_SHA256 or module_state_sha256(decoder) != DECODER_STATE_SHA256:
        raise RuntimeError('base Flow/mesh decoder differs from the paper frozen tensors')
    base.eval().requires_grad_(False)
    base.use_checkpoint = False
    for block in base.blocks:
        block.use_checkpoint = False
    flow = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
                                         target_modules=['to_q', 'to_kv', 'to_out', 'to_qkv']))
    model = FrozenSLat30K(flow)
    parameters = {k: v for k, v in model.named_parameters() if v.requires_grad}
    state = saved['ema_trainable_state']
    if set(parameters) != set(state) or len(state) != 298:
        raise RuntimeError('SLat EMA parameter names differ')
    with torch.no_grad():
        for name, value in state.items():
            if parameters[name].shape != value.shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError('invalid SLat EMA tensor: ' + name)
            parameters[name].copy_(value)
    norm = saved['model_summary']['slat_normalization']
    if canonical_json_sha256(norm) != '7a2b57bebb8aa55f49215fd897ef0cec497ac20a5a2df85576343a50a6fd3c9d':
        raise RuntimeError('SLat normalization differs')
    params = dict(saved['model_summary']['sampler_defaults'], guidance_rescale=0.)
    if params != SAMPLER_PARAMS:
        raise RuntimeError('SLat sampler differs')
    binding = dict(checkpoint=str(checkpoint), sha256=SLAT_SHA256, step=30000, weights='ema',
        loaded_tensor_count=len(state), trainable_parameters_at_inference=0,
        flow=flow_binding, decoder=decoder_binding, sampler=SAMPLER_PARAMS, normalization=norm,
        evaluation_uid_count=len(evaluation_uids), slat_training_disjoint=True,
        training_uid_count=len(training), frame='native_sparse_grid_identity_no_transform_pose')
    binding['determinism'] = dict(torch_deterministic_algorithms=True,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        sparse_downsample='torch_scatter_reduce_mean_original_include_self_true',
        weights_and_forward_equations_unchanged=True)
    model.eval().requires_grad_(False).to(device)
    decoder.eval().requires_grad_(False).to(device)
    return dict(model=model, decoder=decoder, sampler=samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1.e-5),
        binding=binding, mean=torch.tensor(norm['mean'], device=device)[None],
        std=torch.tensor(norm['std'], device=device)[None])


@torch.no_grad()
def sample_mesh_latent(runtime, sample, initial):
    inputs, condition = inference_inputs(sample, initial.device)
    flow = FrozenSLatCFG(runtime['model'], condition['cond'], inputs)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        latent = runtime['sampler'].sample(flow, initial, **condition, **SAMPLER_PARAMS, verbose=False).samples
    latent = latent * runtime['std'] + runtime['mean']
    if not torch.equal(latent.coords, initial.coords) or not bool(torch.isfinite(latent.feats).all()):
        raise RuntimeError('invalid SLat sample')
    return latent, dict(positive_calls=flow.positive_calls, negative_calls=flow.negative_calls)
