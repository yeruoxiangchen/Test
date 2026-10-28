import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch

from ...tests.common import synthetic_sample
from ...tests.test_native_ss_conditions import DummySSFlow
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import JointSSFlow
from ...contracts import ground_truth_prediction
from .model import NestedSS, ResidualAdapter, observations
from .common import ARMS, tensor_hash
from .train import fixed_batch, fixed_noise, recover_history


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def base(self):
        config = GeometryConfig(hidden_channels=16, evidence_channels=4, candidates=4, query_chunk=5)
        return JointSSFlow(DummySSFlow(), config, condition_mode='no_layer')

    def test_nested_preparation_preserves_both_appearance_conditions(self):
        sample = synthetic_sample(channels=1024)
        base = self.base()
        oracle = ground_truth_prediction(sample)
        with torch.no_grad(), patch.object(base, 'predict_layers', return_value=oracle) as predict:
            a, l, _ = base.build_conditions(sample, device='cpu')
            hashes = []
            for arm in ARMS:
                model = NestedSS(base, arm)
                prepared = model.prepare(sample, 'cpu')
                self.assertTrue(torch.equal(a, prepared['appearance']))
                self.assertTrue(torch.equal(l, prepared['appearance_layer']))
                hashes.append(tensor_hash(model.residual.state_dict()))
                self.assertTrue(all(n.startswith('residual.') for n, p in model.named_parameters() if p.requires_grad))
            self.assertEqual(len(set(hashes)), 1)
            supplied = predict.call_args.args[0]
            self.assertNotIn('target', supplied)
            self.assertNotIn('layer_depths_gt', supplied)

    def test_prediction_works_without_gt_and_oracle_requires_gt(self):
        sample = synthetic_sample(channels=1024)
        pred = ground_truth_prediction(sample)
        clean = observations(sample)
        base = self.base()
        with patch.object(base, 'predict_layers', return_value=pred):
            self.assertEqual(NestedSS(base, 'predicted').prepare(clean, 'cpu')['features'].shape[-1], 12)
        with self.assertRaises((KeyError, ValueError)):
            NestedSS(base, 'oracle').prepare(clean, 'cpu')

    def test_zero_init_gradient_flow_and_invalid_frustum(self):
        adapter = ResidualAdapter(8, 2, channels=16)
        features, weights = torch.randn(2, 10, 12), torch.full((2, 10), .5)
        valid = torch.ones(2, 10, dtype=torch.bool)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001)
        target = torch.randn(1, 10, 8)
        for step in range(2):
            optimizer.zero_grad()
            z = adapter.encode(features, weights, valid)
            output = sum(p(z) for p in adapter.projections)
            if step == 0:
                self.assertEqual(int(output.count_nonzero()), 0)
            (output - target).square().mean().backward()
            self.assertGreater(sum(float(p.weight.grad.abs().sum()) for p in adapter.projections), 0.)
            if step:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in adapter.encoder.parameters()), 0.)
            optimizer.step()
        invalid = adapter.encode(features, weights, torch.zeros_like(valid))
        self.assertEqual(int(invalid.count_nonzero()), 0)
        self.assertTrue(all(int(p(invalid).count_nonzero()) == 0 for p in adapter.projections))

    def test_stateless_noise_and_order_are_resume_and_cache_independent(self):
        target = torch.zeros(1, 8, 4, 4, 4)
        a, x = fixed_noise(target, 41, 19)
        torch.manual_seed(999)
        torch.randn(1000)
        b, y = fixed_noise(target, 41, 19)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(fixed_batch(range(16), 41, 19), fixed_batch(range(16), 41, 19))
        self.assertEqual(len({fixed_batch(range(16), 41, i)[0] for i in range(16)}), 16)

    def test_interrupted_history_replays_only_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.jsonl'
            path.write_text('{"step": 1}\n{"step": 2}\n{"step":')
            recover_history(path, 1)
            self.assertEqual(path.read_text(), '{"step": 1}\n')
            self.assertEqual(len(list(Path(folder).glob('history_interrupted_*.jsonl'))), 1)
            with self.assertRaisesRegex(ValueError, 'disagree'):
                recover_history(path, 2)


if __name__ == '__main__':
    unittest.main()
