"""V2 SS extension; the frozen v1 source and checkpoint contracts stay intact."""
import torch

from ..native_ss import build_layered_ss_components, LayeredNativeSSFlow
from ..checkpoints import load_parent_ss30k
from ..contracts import ground_truth_prediction
from ..projection import build_ss_projection, project_masked_dino, canonical_query_points
from ..view_fusion import aggregate_layer_evidence
from .layer_predictor import GeometryConfig, RayLayerPredictor
from .conditioning import (CONDITION_MODES, first_surface_prediction,
                           projected_appearance_inputs, query_geometry)
from .contracts import LayeredPredictionV2


class JointSSFlow(LayeredNativeSSFlow):
    def __init__(self, flow, config, absolute_tolerance=None, condition_mode='full'):
        super().__init__(flow, lari_layers=config.layers)
        if condition_mode not in CONDITION_MODES:
            raise ValueError(f'unknown SS condition mode: {condition_mode}')
        self.condition_mode = condition_mode
        self.target_lari = RayLayerPredictor(config)
        # Instantiate after shared adapters: a fixed seed gives identical shared
        # adapter initialization for every condition mode.  This small extra
        # projection is a real control parameter, never counted as an unused
        # capacity-matching module in full/single-layer models.
        if condition_mode == 'no_layer':
            self.appearance_control = torch.nn.Sequential(
                torch.nn.Linear(2 * self.condition_channels + 1, 12), torch.nn.SiLU())
        self.match_absolute_tolerance = config.absolute_tolerance if absolute_tolerance is None else absolute_tolerance
        if self.match_absolute_tolerance <= 0:
            raise ValueError('SS matching tolerance must be positive')

    def build_conditions(self, sample, *, device, view_indices=None, use_layer_branch=True,
                         layer_source='predicted', layer_prediction=None, projection_mode='correct',
                         **diagnostics):
        if view_indices is not None:
            raise ValueError('select all input/context views together with v2.dataset.select_views first')
        if projection_mode != 'correct' or any(value != 'correct' for value in diagnostics.values()):
            raise ValueError('V2 diagnostics require explicit modified inputs, not v1 corruption flags')
        appearance_projection = build_ss_projection(sample, device=device)
        appearance = project_masked_dino(sample, appearance_projection, device=device)
        if not use_layer_branch:
            aggregated, stats, _ = self.aggregator(appearance, appearance_projection['valid'])
            return aggregated, None, stats
        if self.condition_mode == 'no_layer':
            if layer_prediction is not None or layer_source != 'predicted':
                raise ValueError('no_layer conditions do not accept predicted or oracle layer tensors')
            features = self.appearance_control(projected_appearance_inputs(
                sample, appearance_projection, appearance, device=device))
            residual = self.view_gate(features)
            condition, stats, details = self.aggregator(appearance, appearance_projection['valid'],
                                                        logit_residual=residual)
            support = appearance_projection['valid'].float()
            layer, layer_stats = aggregate_layer_evidence(self.layer_encoder(features),
                appearance_weights=details['view_weights'], target_support=support)
            return condition, layer, {**stats, **layer_stats,
                'target_support_mean': support.new_zeros(()),
                'control_frustum_support_mean': support.mean(),
                'view_logit_residual_rms': residual.square().mean().sqrt()}
        prediction = layer_prediction
        if prediction is None:
            if layer_source == 'oracle':
                oracle = ground_truth_prediction(sample).to(device)
                prediction = LayeredPredictionV2(oracle.depths, oracle.stop_logits, oracle.target_logits, oracle.confidence_logits)
            elif layer_source == 'predicted':
                prediction = self.predict_layers(sample, device=device, view_indices=None, source='predicted')
            else:
                raise ValueError('layer source must be predicted or oracle')
        if self.condition_mode == 'single_layer':
            prediction = first_surface_prediction(prediction)
        points = canonical_query_points(16, sample['grid_transform'], device=device)
        config = self.target_lari.config
        match = query_geometry(sample, prediction, points,
            absolute_tolerance=self.match_absolute_tolerance, relative_tolerance=config.relative_tolerance)
        residual = self.view_gate(match.features)
        condition, stats, details = self.aggregator(appearance, appearance_projection['valid'], logit_residual=residual)
        layer, layer_stats = aggregate_layer_evidence(self.layer_encoder(match.features),
            appearance_weights=details['view_weights'], target_support=match.target_support)
        return condition, layer, {**stats, **layer_stats,
            'target_support_mean': match.target_support.mean(),
            'view_logit_residual_rms': residual.square().mean().sqrt()}

    def adapted_prediction(self, x, t, condition, sample, *, prepared_conditions=None, **kwargs):
        if prepared_conditions is None or sample is None:
            return super().adapted_prediction(x, t, condition, sample, **kwargs)
        appearance, layer, stats = prepared_conditions
        prediction = self._core_forward(x, t, condition, appearance, layer)
        return prediction, {**stats, 'condition_present': x.new_tensor(1.), 'layer_branch_present': x.new_tensor(1.)}


def build_ss(config, device, parent_checkpoint, pretrained='Stable-X/trellis-vggt-v0-2', gradient_checkpointing=True,
             absolute_tolerance=None, condition_mode='full'):
    sampler, base, summary = build_layered_ss_components(pretrained=pretrained, device=device,
        gradient_checkpointing=gradient_checkpointing, lari_layers=config.layers)
    model = JointSSFlow(base.flow, config, absolute_tolerance, condition_mode).to(device)
    model.stock_binding = base.stock_binding
    del base
    parent = load_parent_ss30k(model, parent_checkpoint)
    if condition_mode == 'no_layer':
        # The legacy parent loader freezes names outside its layered namespace.
        model.appearance_control.requires_grad_(True)
    return sampler, model, {'parent': parent, 'stock': summary['stock_content_identity']}
