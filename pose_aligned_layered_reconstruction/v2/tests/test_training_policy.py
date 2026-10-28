import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from ...tests.common import synthetic_sample
from ...tests.test_native_ss_conditions import DummySSFlow
from ..checkpoints import initialize_ema, load_geometry, load_state, save_checkpoint, read_checkpoint
from ..layer_predictor import GeometryConfig, RayLayerPredictor
from ..ss import JointSSFlow
from ..train import parser, step_loss
from ..training_policy import learning_rate_factor, sample_time, validate_resume_policy


class TrainingPolicyTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--stage', 'ss', '--manifest', 'unused', '--output_dir', 'unused', *extra])

    def test_legacy_resume_defaults_and_policy_changes(self):
        current = vars(self.args())
        old = {k: v for k, v in current.items() if k not in
               ('init_weights', 'lr_schedule', 't_sampling', 'lr_schedule_steps', 'ss_absolute_tolerance')}
        validate_resume_policy(old, current)
        for key, value in [('init_weights', 'raw'), ('lr_schedule', 'constant'),
                           ('t_sampling', 'logit_normal'), ('ss_absolute_tolerance', .04)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resume_policy(old, {**current, key: value})

    def test_pilot_prefix_has_identical_learning_rates(self):
        full = self.args('--max_steps', '2000', '--warmup_steps', '50')
        pilot = self.args('--max_steps', '500', '--lr_schedule_steps', '2000', '--warmup_steps', '50')
        self.assertEqual([learning_rate_factor(full, i) for i in range(1, 501)],
                         [learning_rate_factor(pilot, i) for i in range(1, 501)])
        constant = self.args('--lr_schedule', 'constant')
        self.assertEqual([learning_rate_factor(constant, i) for i in (1, 500, 2000)], [1., 1., 1.])

    def test_time_samplers_reproduce_original_draws(self):
        for policy in ('uniform', 'logit_normal'):
            torch.manual_seed(73)
            actual = sample_time(policy, 'cpu')
            torch.manual_seed(73)
            expected = torch.rand(()).clamp(.001, .999) if policy == 'uniform' else torch.sigmoid(torch.randn(()) + 1.)
            self.assertEqual(actual, float(expected))

    def test_initialization_selects_raw_or_ema_and_tolerance_is_bound(self):
        config = GeometryConfig(input_channels=8, hidden_channels=8, evidence_channels=4, candidates=4)
        model = RayLayerPredictor(config)
        ema = initialize_ema(model)
        with torch.no_grad():
            next(model.parameters()).add_(.25)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'lari.pt'
            save_checkpoint(path, model=model, config=config, stage='lari', bindings={}, data_identity={},
                optimizer=torch.optim.AdamW(model.parameters()), ema=ema, step=1,
                cursor={'epoch': 0, 'offset': 1}, args=argparse.Namespace())
            raw, average = RayLayerPredictor(config), RayLayerPredictor(config)
            load_geometry(raw, path, config, ema=False)
            load_geometry(average, path, config, ema=True)
            self.assertTrue(torch.equal(next(raw.parameters()), next(model.parameters())))
            self.assertTrue(torch.equal(next(average.parameters()), next(iter(ema.values()))))
        joint = JointSSFlow(DummySSFlow(), config, absolute_tolerance=.04)
        cp = {'geometry_config': asdict(config), 'stage': 'ss', 'bindings': {},
              'args': {'ss_absolute_tolerance': .02}}
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            load_state(joint, cp, config, stage='ss', bindings={})

    def test_cached_geometry_preserves_loss_and_adapter_gradients(self):
        torch.set_num_threads(2)
        sample = synthetic_sample(channels=1024)
        sample['stock_condition'] = torch.zeros((1, 1, 1024))
        config = GeometryConfig(hidden_channels=8, evidence_channels=4, candidates=4, query_chunk=8)
        model = JointSSFlow(DummySSFlow(), config)
        model.target_lari.eval().requires_grad_(False)
        # Exercise real geometry/matching/encoders, with a tiny differentiable flow stand-in.
        model.adapted_prediction = lambda x, t, c, s, prepared_conditions: (
            x + prepared_conditions[1].square().mean(), {})
        class Sampler:
            def _get_model_gt(self, target, t, noise):
                return target * (1-t) + noise * t, noise - target
        args = self.args('--p_uncond', '0', '--freeze_geometry', '--cache_frozen_geometry')
        cache = {}
        torch.manual_seed(1)
        loss, _ = step_loss(args, model, Sampler(), sample, torch.device('cpu'), {}, cache)
        loss.backward()
        before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        def forbid_prediction(*a, **kw):
            raise AssertionError('frozen geometry recomputed')
        model.target_lari.predict = forbid_prediction
        torch.manual_seed(1)
        repeat, _ = step_loss(args, model, Sampler(), sample, torch.device('cpu'), {}, cache)
        repeat.backward()
        self.assertTrue(torch.equal(loss, repeat))
        self.assertGreater(len(before), 0)
        for name, p in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(before[name], p.grad), name)


if __name__ == '__main__':
    unittest.main()
