import unittest

import numpy as np
import torch

from .targets import input_target, oracle_targets, novel_cameras
from .optimize import terms
from .metrics import fixed_depth_error
from .report import object_summary


def render(depth=2., views=4, side=4):
    d = torch.full((views, side, side), depth)
    n = torch.zeros((views, side, side, 3)); n[..., 2] = 1
    return dict(depth=d, mask=torch.ones_like(d, dtype=torch.bool), softmask=torch.ones_like(d),
        normal=n, normal_valid=torch.ones_like(d, dtype=torch.bool))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.base = render()
        self.w = torch.eye(4)[None].repeat(4, 1, 1)
        self.pixels = np.array([[0, 1, 1], [2, 2, 2]])
        self.input = input_target(self.base, self.pixels, [[0, 0, 2.2], [0, 0, 1.8]],
            torch.ones_like(self.base['depth']), self.w)
        self.record = dict(target=self.input, pixels=self.pixels, observed=self.input['softmask'])

    def test_oracle_replaces_only_same_pixel_queries_and_preserves_input(self):
        before = {k: v.clone() for k, v in self.input.items()}
        oracle, stats = oracle_targets(self.base, render(2.1), self.record)
        self.assertTrue(torch.equal(oracle['oracle_sparse']['depth_valid'], self.input['depth_valid']))
        for v, y, x in self.pixels:
            self.assertAlmostEqual(float(oracle['oracle_sparse']['depth'][v, y, x]), 2.1, places=5)
        self.assertEqual(stats['query_count'], 2)
        self.assertTrue(stats['pure_paired_depth_comparison_valid'])
        for key in self.input: self.assertTrue(torch.equal(before[key], self.input[key]))

    def test_no_gt_intersection_is_counted_and_never_invents_depth(self):
        gt = render(2.1); gt['mask'][0, 1, 1] = False; gt['depth'][0, 1, 1] = 0
        oracle, stats = oracle_targets(self.base, gt, self.record)
        self.assertEqual(stats['no_gt_intersection'], 1)
        self.assertFalse(bool(oracle['oracle_sparse']['depth_valid'][0, 1, 1]))
        self.assertFalse(stats['pure_paired_depth_comparison_valid'])
        self.assertTrue(bool(self.input['depth_valid'][0, 1, 1]))

    def test_sparse_and_depth_only_arms_have_no_fabricated_normals(self):
        oracle, _ = oracle_targets(self.base, render(2.1), self.record)
        self.assertFalse(bool(oracle['oracle_sparse']['normal_valid'].any()))
        self.assertFalse(bool(oracle['oracle_dense_depth']['normal_valid'].any()))
        self.assertTrue(bool(oracle['oracle_dense_normal']['normal_valid'].all()))

    def test_eval_or_heldout_views_never_enter_training_loss(self):
        gt = render(2.1); changed = render(2.1); changed['depth'][1] = 50
        region = torch.ones_like(gt['depth'])
        a = terms(self.base, gt, self.base, region, [0, 2, 3])
        b = terms(self.base, changed, self.base, region, [0, 2, 3])
        self.assertEqual(float(a['loss']), float(b['loss']))

    def test_frozen_missing_ray_domain_cannot_shrink_away(self):
        pred = render(2.2); gt = render(2.1)
        a = terms(pred, gt, self.base, torch.ones_like(gt['depth']), [0])
        pred['mask'][0] = False
        b = terms(pred, gt, self.base, torch.ones_like(gt['depth']), [0])
        self.assertGreaterEqual(float(b['local_depth']), float(a['local_depth']) - 1.e-5)

    def test_novel_cameras_are_fixed_orthonormal_and_separate_from_inputs(self):
        w = np.repeat(np.eye(4)[None], 4, 0)
        w[:, 2, 3] = 2
        k = np.repeat(np.eye(3)[None], 4, 0)
        a, b, receipt = novel_cameras(k, w)
        a2, b2, _ = novel_cameras(k, w)
        np.testing.assert_array_equal(a, a2); np.testing.assert_array_equal(b, b2)
        for rotation in b[:, :3, :3]:
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.e-6)
            self.assertAlmostEqual(np.linalg.det(rotation), 1., places=5)
        self.assertGreaterEqual(receipt['minimum_input_angle_degrees'], 10)

    def test_fixed_evaluation_penalizes_lost_geometry_and_false_positive_surface(self):
        gt, pred = render(), render()
        pred['mask'][0, 1, 1] = False
        gt['mask'][0, 2, 2] = False
        error = fixed_depth_error(pred, gt)
        self.assertAlmostEqual(float(error[0, 1, 1]), .1, places=6)
        self.assertAlmostEqual(float(error[0, 2, 2]), .1, places=6)
        self.assertEqual(float(error[0, 0, 0]), 0)

    def test_statistics_average_objects_before_seeds_and_do_not_fill_missing(self):
        value = object_summary([(1, 42, 1.), (1, 43, 3.), (2, 42, 100.), (2, 43, None)])
        self.assertEqual(value['mean'], 51.)
        self.assertEqual(value['objects'], 2)
        self.assertEqual(value['cases'], 3)


if __name__ == '__main__': unittest.main()
