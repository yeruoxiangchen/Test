"""Frozen trained appearance base plus a separate, zero-init SS residual."""
import torch
from torch import nn
import torch.nn.functional as F

from ...native_ss import volume_xyz_to_flow_tokens, flow_tokens_to_volume_xyz
from ..._bootstrap import bootstrap_external_paths
from ...projection import build_ss_projection, project_masked_dino, canonical_query_points
from ...view_fusion import aggregate_layer_evidence
from ...contracts import ground_truth_prediction
from ...v2.conditioning import projected_appearance_inputs, query_geometry
from ...v2.contracts import LayeredPredictionV2
from ...v2.checkpoints import read_checkpoint, load_state
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import build_ss
from .common import ARMS, SEED, verify
from .query import geometry_features

OBSERVATIONS = ('format', 'uid', 'object_uid', 'raw_visual_patch_features', 'masked_visual_patch_features',
    'visible_target_mask', 'intrinsics', 'extrinsics', 'image_size', 'extrinsics_type',
    'camera_forward_sign', 'layer_camera_y_sign', 'grid_transform')


def observations(sample):
    return {k: sample[k] for k in OBSERVATIONS if k in sample}


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class ResidualAdapter(nn.Module):
    def __init__(self, width, blocks, channels=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.norm = nn.LayerNorm(channels)
        self.projections = nn.ModuleList(nn.Linear(channels, width, bias=False) for _ in range(blocks))
        for projection in self.projections:
            nn.init.zeros_(projection.weight)

    def encode(self, features, weights, valid):
        # Reliability is retained in the 12 input features, not multiplied into
        # amplitude a second time. No division by tiny target-support mass.
        encoded = self.encoder(features)
        fused = (encoded.float() * weights[..., None].float() * valid[..., None]).sum(0)
        return (self.norm(fused) * valid.any(0)[:, None])[None]


class NestedSS(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.base = base.eval().requires_grad_(False)
        self.arm = arm
        # Identical adapter initialization, independent of base-model load RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.residual = ResidualAdapter(base.flow_core.model_channels, len(base.flow_core.blocks))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def prepare(self, sample, device):
        obs = observations(sample)
        projection = build_ss_projection(obs, device=device)
        appearance = project_masked_dino(obs, projection, device=device)
        control = self.base.appearance_control(projected_appearance_inputs(obs, projection, appearance, device=device))
        parent, stats, details = self.base.aggregator(appearance, projection['valid'],
                                                    logit_residual=self.base.view_gate(control))
        parent_layer, _ = aggregate_layer_evidence(self.base.layer_encoder(control),
            appearance_weights=details['view_weights'], target_support=projection['valid'].float())
        if self.arm == 'appearance_control':
            features = control
        else:
            if self.arm.startswith('oracle_'):
                # Only this explicitly privileged arm can access labels.
                oracle = ground_truth_prediction(sample).to(device)
                prediction = LayeredPredictionV2(oracle.depths, oracle.stop_logits,
                                                 oracle.target_logits, oracle.confidence_logits)
            else:
                prediction = self.base.predict_layers(obs, device=device, view_indices=None, source='predicted')
            points = canonical_query_points(16, obs['grid_transform'], device=device)
            features = geometry_features(obs, prediction, points,
                kind='cell_mean_27' if self.arm.endswith('_cell') else 'point')
        return {'appearance': parent.detach(), 'appearance_layer': parent_layer.detach(),
                'features': features.detach(), 'weights': details['view_weights'].detach(),
                'valid': projection['valid'].detach()}

    def forward(self, x, t, context, prepared, enabled=True):
        if not enabled:
            return self.base._core_forward(x, t, context, prepared['appearance'], prepared['appearance_layer'])
        bootstrap_external_paths()
        from trellis.modules.spatial import patchify, unpatchify
        layer = self.residual.encode(prepared['features'], prepared['weights'], prepared['valid'])
        core = self.base.flow_core
        h = volume_xyz_to_flow_tokens(patchify(x, core.patch_size))
        h = core.input_layer(h) + core.pos_emb[None]
        embedding = core.t_embedder(t)
        if core.share_mod:
            embedding = core.adaLN_modulation(embedding)
        h, embedding = h.type(core.dtype), embedding.type(core.dtype)
        for ctx in context if isinstance(context, list) else [context]:
            ctx = ctx.type(core.dtype)
            for i, block in enumerate(core.blocks):
                # Preserve the exact ordering of both original appearance additions.
                h = h + self.base.block_condition(i, prepared['appearance']).to(h.dtype)
                h = h + self.base.layer_block_condition(i, prepared['appearance_layer']).to(h.dtype)
                h = h + self.residual.projections[i](layer).to(h.dtype)
                h = block(h, embedding, ctx)
        h = F.layer_norm(h.type(x.dtype), h.shape[-1:])
        return unpatchify(flow_tokens_to_volume_xyz(core.out_layer(h)), core.patch_size).contiguous()

    @torch.no_grad()
    def metrics(self, prepared):
        layer = self.residual.encode(prepared['features'], prepared['weights'], prepared['valid'])
        return {'residual_condition_rms': rms(layer),
            'residual_injection_rms': sum(rms(p(layer)) for p in self.residual.projections) / len(self.residual.projections),
            'input_feature_rms': rms(prepared['features']),
            'geometry_support_mean': float(prepared['features'][..., 0].float().mean()) if self.arm != 'appearance_control' else None}


def build(contract, device, checkpointing=False):
    checkpoint = read_checkpoint(verify(contract['appearance_base']))
    saved = checkpoint['args']
    if saved['condition_mode'] != 'no_layer' or not saved['freeze_geometry']:
        raise ValueError('Residual base must be the trained, frozen-geometry no_layer model')
    config = GeometryConfig(**checkpoint['geometry_config'])
    sampler, base, bindings = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'],
        checkpointing, absolute_tolerance=saved.get('ss_absolute_tolerance'), condition_mode='no_layer')
    base.target_lari.requires_grad_(False)
    load_state(base, checkpoint, config, stage='ss', bindings=bindings, ema=False)
    model = NestedSS(base, contract['arm']).to(device)
    return sampler, model, saved, bindings


class ResidualCFG(nn.Module):
    def __init__(self, model, positive, prepared, enabled=True):
        super().__init__()
        self.model, self.positive, self.prepared, self.enabled = model, positive, prepared, enabled
        self.positive_calls = self.negative_calls = 0

    def forward(self, x, t, condition):
        positive = condition is self.positive or (torch.is_tensor(condition) and condition.data_ptr() == self.positive.data_ptr())
        if positive:
            self.positive_calls += 1
            return self.model(x, t, condition, self.prepared, self.enabled)
        self.negative_calls += 1
        return self.model.base.adapted_prediction(x, t, condition, None)[0]
