"""Causal controls: no geometry leaks, first-surface isolation and exact resume."""
import argparse
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from ...contracts import ground_truth_prediction
from ...tests.common import synthetic_sample
from ...tests.test_native_ss_conditions import DummySSFlow
from ..checkpoints import initialize_ema, load_state, read_checkpoint, save_checkpoint
from ..conditioning import first_surface_prediction
from ..contracts import LayeredPredictionV2
from ..layer_predictor import GeometryConfig
from ..ss import JointSSFlow, build_ss
from ..train import parameter_inventory, parser, step_loss


class FairConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = GeometryConfig(hidden_channels=8, evidence_channels=4, candidates=4, query_chunk=8)
        self.sample = synthetic_sample(channels=1024)
        self.sample['stock_condition'] = torch.zeros((1, 1, 1024))

    def model(self, mode):
        torch.manual_seed(31)
        model = JointSSFlow(DummySSFlow(), self.config, condition_mode=mode)
        model.target_lari.requires_grad_(False).eval()
        model.flow.requires_grad_(False)
        model.aggregator.requires_grad_(False)
        model.block_condition.requires_grad_(False)
        return model

    def test_single_surface_preserves_first_existence_without_later_depths(self):
        pred = ground_truth_prediction(self.sample)
        first = first_surface_prediction(pred)
        self.assertEqual(first.depths.shape[1], 1)
        torch.testing.assert_close(first.valid_probabilities(), pred.valid_probabilities()[:, :1])
        altered = LayeredPredictionV2(pred.depths.clone(), pred.stop_logits.clone(),
                                     pred.target_logits.clone(), pred.confidence_logits.clone())
        altered.depths[:, 1:] += 11.
        altered.target_logits[:, 1:] -= 21.
        altered.confidence_logits[:, 1:] -= 15.
        other = first_surface_prediction(altered)
        for name in ('depths', 'stop_logits', 'target_logits', 'confidence_logits'):
            self.assertTrue(torch.equal(getattr(first, name), getattr(other, name)), name)
        model = self.model('single_layer')
        before = model.build_conditions(self.sample, device=torch.device('cpu'), layer_prediction=pred)
        after = model.build_conditions(self.sample, device=torch.device('cpu'), layer_prediction=altered)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_no_layer_reads_observations_and_never_calls_geometry_or_labels(self):
        model = self.model('no_layer')
        sample = {k: v for k, v in self.sample.items() if not k.endswith('_gt') and k != 'target'}
        def forbidden(*args, **kwargs):
            raise AssertionError('no_layer tried to predict layers')
        model.predict_layers = forbidden
        model.target_lari.predict = forbidden
        observed = []
        hook = model.appearance_control[0].register_forward_pre_hook(lambda module, args: observed.append(args[0]))
        parent = model.build_conditions(sample, device=torch.device('cpu'), use_layer_branch=False)[0]
        appearance, layer, _ = model.build_conditions(sample, device=torch.device('cpu'))
        self.assertTrue(torch.equal(parent, appearance))
        self.assertTrue(model.new_adapters_exact_zero())
        self.assertEqual(observed[0].shape[-1], 2049)
        altered = {**sample, 'raw_visual_patch_features': sample['raw_visual_patch_features'] + 3.}
        changed = model.build_conditions(altered, device=torch.device('cpu'))[1]
        self.assertGreater(float((changed - layer).abs().max()), 0.)
        layer.square().mean().backward()
        self.assertGreater(float(model.appearance_control[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.target_lari.parameters()))
        hook.remove()
        with self.assertRaisesRegex(ValueError, 'do not accept'):
            model.build_conditions(sample, device=torch.device('cpu'), layer_source='oracle')

    def test_all_shared_adapter_initialization_and_diffusion_rng_match(self):
        models = {mode: self.model(mode) for mode in ('full', 'single_layer', 'no_layer')}
        shared = dict(models['full'].named_parameters())
        reference_capacity, reference_hashes = parameter_inventory(models['full'])
        for model in models.values():
            capacity, hashes = parameter_inventory(model)
            self.assertEqual(reference_hashes, hashes)
            extra = 24600 if model.condition_mode == 'no_layer' else 0
            self.assertEqual(reference_capacity['trainable'] + extra, capacity['trainable'])
            for name, param in model.named_parameters():
                if name in shared:
                    self.assertTrue(torch.equal(shared[name], param), name)
        draws = []
        class Sampler:
            def _get_model_gt(self, target, t, noise):
                draws.append((t, noise.clone()))
                return target * (1-t) + noise * t, noise - target
        for mode, model in models.items():
            model.adapted_prediction = lambda x, t, c, s, prepared_conditions: (
                x + prepared_conditions[1].square().mean(), {})
            args = parser().parse_args(['--stage', 'ss', '--manifest', 'unused', '--output_dir', 'unused',
                '--condition_mode', mode, '--freeze_geometry', '--p_uncond', '0'])
            random.seed(19)
            torch.manual_seed(19)
            loss, metrics = step_loss(args, model, Sampler(), self.sample, torch.device('cpu'), {})
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            if mode == 'no_layer':
                self.assertNotIn('layer_loss', metrics)
        for t, noise in draws[1:]:
            self.assertEqual(t, draws[0][0])
            self.assertTrue(torch.equal(noise, draws[0][1]))

    def test_condition_mode_is_bound_to_checkpoint_and_legacy_defaults_full(self):
        for mode in ('full', 'single_layer', 'no_layer'):
            model = self.model(mode)
            args = argparse.Namespace(condition_mode=mode, ss_absolute_tolerance=None)
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'joint.pt'
                save_checkpoint(path, model=model, config=self.config, stage='ss', bindings={},
                    data_identity={}, optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]),
                    ema=initialize_ema(model), step=1, cursor={'epoch': 0, 'offset': 1}, args=args)
                checkpoint = read_checkpoint(path)
                self.assertEqual(checkpoint['condition_mode'], mode)
                clone = self.model(mode)
                load_state(clone, checkpoint, self.config, stage='ss', bindings={})
                changed = self.model('single_layer' if mode == 'full' else 'full')
                with self.assertRaisesRegex(ValueError, 'condition_mode'):
                    load_state(changed, checkpoint, self.config, stage='ss', bindings={})
                if mode == 'full':
                    checkpoint.pop('condition_mode')
                    checkpoint['args'].pop('condition_mode')
                    load_state(clone, checkpoint, self.config, stage='ss', bindings={})

    def test_parent_loader_freeze_does_not_disable_appearance_control(self):
        base = self.model('full')
        base.stock_binding = {}
        def frozen_parent(model, path):
            allowed = ('target_lari.', 'layer_encoder.', 'view_gate.', 'layer_block_condition.')
            for name, param in model.named_parameters():
                param.requires_grad_(name.startswith(allowed))
            return {}
        with patch('pose_aligned_layered_reconstruction.v2.ss.build_layered_ss_components',
                   return_value=(None, base, {'stock_content_identity': {}})), \
             patch('pose_aligned_layered_reconstruction.v2.ss.load_parent_ss30k', side_effect=frozen_parent):
            _, model, _ = build_ss(self.config, torch.device('cpu'), 'unused', condition_mode='no_layer')
        self.assertTrue(all(p.requires_grad for p in model.appearance_control.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.aggregator.parameters()))


if __name__ == '__main__':
    unittest.main()
