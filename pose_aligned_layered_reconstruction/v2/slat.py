"""Trainable geometry adapters around the hash-verified SLat30K parent."""
import torch
import os
import torch.nn.functional as F
from torch import nn

from ..frozen_slat import FrozenSLat30K, load_frozen_slat30k, project_active_dino
from ..view_fusion import LayerEvidenceEncoder, EveryBlockConditionProjection, aggregate_layer_evidence
from .conditioning import active_points, query_geometry
from .layer_predictor import RayLayerPredictor
from .ray_geometry import CameraBundle


def predict_geometry(geometry, sample, device):
    cameras = CameraBundle.from_sample(sample, device)
    return geometry(raw_patch_features=sample['raw_visual_patch_features'].to(device).float(),
        masked_patch_features=sample['masked_visual_patch_features'].to(device).float(),
        visible_target_mask=sample['visible_target_mask'].to(device).float(),
        intrinsics=cameras.intrinsics, extrinsics=cameras.w2c, extrinsics_type='w2c',
        camera_forward_sign=cameras.forward_sign, image_size=cameras.image_size)


class GeometrySLat(FrozenSLat30K):
    def __init__(self, parent, geometry, layer_channels=128):
        # Preserve parent modules and tensors, including both CFG-branch biases.
        nn.Module.__init__(self)
        self.flow = parent.flow
        self.aggregator = parent.aggregator
        self.block_condition = parent.block_condition
        self.geometry = geometry.eval().requires_grad_(False)
        self.layer_encoder = LayerEvidenceEncoder(output_channels=layer_channels)
        self.layer_block_condition = EveryBlockConditionProjection(layer_channels,
            int(self.flow_core.model_channels), len(self.flow_core.blocks))

    def forward(self, x, t, condition, sample, cache):
        core = self.flow_core
        h = core.input_layer(x).type(core.dtype)
        time_embedding = core.t_embedder(t)
        if core.share_mod:
            time_embedding = core.adaLN_modulation(time_embedding)
        time_embedding = time_embedding.type(core.dtype)
        condition = [v.type(core.dtype) for v in condition] if isinstance(condition, list) else condition.type(core.dtype)
        skips = []
        for block in core.input_blocks:
            h = block(h, time_embedding)
            skips.append(h.feats)
        if core.pe_mode == 'ape':
            h = h + core.pos_embedder(h.coords[:, 1:]).type(core.dtype)
        if sample is not None:
            if not cache:
                projected, valid = project_active_dino(sample, h.coords, x.device)
                aggregated, _, _ = self.aggregator(projected, valid)
                cache.update(coords=h.coords.clone(), condition=aggregated[0])
            if not torch.equal(cache['coords'], h.coords):
                raise RuntimeError('SLat stem coordinate order changed')
            appearance = cache['condition']
        else:
            appearance = x.feats.new_zeros((len(h.feats), 1024), dtype=torch.float32)
        for i, block in enumerate(core.blocks):
            residual = self.block_condition(i, appearance)
            extra = self.extra_block_condition(i, h, sample, cache)
            if extra is not None:
                residual = residual + extra
            h = block(h.replace(h.feats + residual.to(h.dtype)), time_embedding, condition)
        for block, skip in zip(core.out_blocks, reversed(skips)):
            if core.use_skip_connection:
                h = block(h.replace(torch.cat((h.feats, skip), dim=1)), time_embedding)
            else:
                h = block(h, time_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return core.out_layer(h.type(x.dtype))

    def extra_block_condition(self, block_index, h, sample, cache):
        if sample is None:
            return None
        if 'layer_condition' not in cache:
            prediction = predict_geometry(self.geometry, sample, h.device)
            points = active_points(h.coords, sample['grid_transform'])
            match = query_geometry(sample, prediction, points,
                absolute_tolerance=self.geometry.config.absolute_tolerance,
                relative_tolerance=self.geometry.config.relative_tolerance)
            projected, valid = project_active_dino(sample, h.coords, h.device)
            _, _, details = self.aggregator(projected, valid)
            layer, _ = aggregate_layer_evidence(self.layer_encoder(match.features),
                appearance_weights=details['view_weights'], target_support=match.target_support)
            cache['layer_condition'] = layer[0]
        return self.layer_block_condition(block_index, cache['layer_condition'])


def build_slat(config, device, geometry_checkpoint, checkpoint, finetune_lora=False, init_ema=True):
    from .checkpoints import load_geometry
    runtime = load_frozen_slat30k(device, checkpoint)
    parent, decoder, sampler, binding = (runtime[k] for k in ('model', 'decoder', 'sampler', 'binding'))
    binding = {**binding, 'v2_sparse_algorithm': os.environ.get('SPCONV_ALGO', 'auto')}
    geometry = RayLayerPredictor(config).to(device)
    load_geometry(geometry, geometry_checkpoint, config, ema=init_ema)
    model = GeometrySLat(parent, geometry).to(device)
    if finetune_lora:
        for name, parameter in model.flow.named_parameters():
            if 'lora_' in name:
                parameter.requires_grad_(True)
    return sampler, model, decoder, binding
