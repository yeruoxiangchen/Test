import unittest
from unittest.mock import patch
import torch
from ...tests.common import synthetic_sample
from ...tests.test_native_ss_conditions import DummySSFlow
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import JointSSFlow
from ...contracts import ground_truth_prediction
from ..hssd_residual_20261021.model import NestedSS as OldNestedSS
from .model import NestedSS, observations
from .query import offsets, geometry_features
from .common import ARMS, tensor_hash
from .validation import choose, CATEGORIES


class CellQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_midpoint_quadrature_geometry_and_rotation(self):
        d = offsets('identity', device='cpu')
        self.assertEqual(d.shape, (27, 3))
        self.assertTrue(torch.allclose(d.mean(0), torch.zeros(3), atol=1e-8, rtol=0.))
        self.assertTrue(bool((d.abs() < 1/32).all()))
        rotation = d.new_tensor(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
        self.assertTrue(torch.equal(offsets('pixal3d_rotation', device='cpu'), d @ rotation.T))
        with self.assertRaises(ValueError): offsets('unknown', device='cpu')

    def test_chunking_and_constant_evidence(self):
        from types import SimpleNamespace
        def fake(sample, pred, points, **kw):
            features = torch.ones(2, len(points), 12)
            features[..., 0] = points[:, 0]
            return SimpleNamespace(features=features)
        sample = {'grid_transform': 'identity'}
        points = torch.randn(13, 3)
        with patch('pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023.query.query_geometry', side_effect=fake):
            a = geometry_features(sample, None, points, kind='cell_mean_27', chunk=2)
            b = geometry_features(sample, None, points, kind='cell_mean_27', chunk=13)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.allclose(a[..., 0], points[None, :, 0].expand(2, -1), atol=1e-6))
            self.assertTrue(torch.equal(a[..., 1:], torch.ones_like(a[..., 1:])))

    def test_old_point_parity_and_cell_gt_whitelist(self):
        sample = synthetic_sample(channels=1024)
        config = GeometryConfig(hidden_channels=16, evidence_channels=4, candidates=4, query_chunk=5)
        base = JointSSFlow(DummySSFlow(), config, condition_mode='no_layer')
        pred = ground_truth_prediction(sample)
        hashes = []
        with patch.object(base, 'predict_layers', return_value=pred) as predict:
            for arm in ARMS:
                model = NestedSS(base, arm)
                supplied = sample if arm.startswith('oracle_') else observations(sample)
                prepared = model.prepare(supplied, 'cpu')
                self.assertEqual(prepared['features'].shape, (len(sample['intrinsics']), 4096, 12))
                self.assertTrue(torch.isfinite(prepared['features']).all())
                old_arm = arm.removesuffix('_point').removesuffix('_cell')
                if not arm.endswith('_cell'):
                    old = OldNestedSS(base, old_arm).prepare(supplied, 'cpu')
                    for key in old: self.assertTrue(torch.equal(old[key], prepared[key]), key)
                hashes.append(tensor_hash(model.residual.state_dict()))
                self.assertTrue(all(n.startswith('residual.') for n,p in model.named_parameters() if p.requires_grad))
            self.assertEqual(len(set(hashes)), 1)
            self.assertNotIn('layer_depths_gt', predict.call_args.args[0])
            self.assertNotIn('target', predict.call_args.args[0])
            with self.assertRaises((KeyError, ValueError)):
                NestedSS(base, 'oracle_cell').prepare(observations(sample), 'cpu')

    def test_selection_disjoint_families_deterministic(self):
        rows = [{'scene': 'v', 'asset_family': str(i//2), 'instance': i, 'category': CATEGORIES[i//2 % len(CATEGORIES)]} for i in range(80)]
        excluded = {'0', '1', '2'}
        chosen = choose(rows, excluded)
        self.assertEqual(chosen, choose(list(reversed(rows)), excluded))
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len({r['asset_family'] for r in chosen}), 16)
        self.assertFalse({r['asset_family'] for r in chosen} & excluded)
        with self.assertRaises(ValueError): choose(rows[:2], excluded)


if __name__ == '__main__': unittest.main()
