import unittest
import torch
from .diagnose import replace_parts, support_from_parts


class CounterfactualTests(unittest.TestCase):
    def test_gt_depth_does_not_copy_invalid_oracle_suffix(self):
        pred = tuple(torch.tensor([[[v, v]]]) for v in (1., .8, .7, .6))
        gt = (torch.tensor([[[2., 999.]]]), torch.tensor([[[1., 0.]]]),
              torch.ones(1, 1, 2), torch.ones(1, 1, 2))
        out, tau = replace_parts(pred, gt, 'gt_depth')
        self.assertEqual(out[0].tolist(), [[[2., 1.]]])
        for i in (1, 2, 3):
            self.assertIs(out[i], pred[i])
        self.assertEqual(tau, .02)
        self.assertEqual(pred[0].tolist(), [[[1., 1.]]])

    def test_confidence_override_preserves_other_components(self):
        pred = tuple(torch.full((2, 3, 5), v) for v in (1., .8, .7, .6))
        out, tau = replace_parts(pred, pred, 'confidence_one')
        for i in (0, 1, 2):
            self.assertIs(out[i], pred[i])
        self.assertTrue(torch.equal(out[3], torch.ones_like(out[3])))
        self.assertTrue(torch.equal(pred[3], torch.full_like(pred[3], .6)))

    def test_frustum_invalid_cannot_get_support_and_analytic_decay(self):
        values = tuple(torch.tensor([[[v], [v]]]) for v in (1., .8, .7, .6))
        projection = {'camera_depth': torch.tensor([[1.02, 1.02]]), 'valid': torch.tensor([[True, False]])}
        support = support_from_parts(values, projection, .02)
        self.assertAlmostEqual(float(support[0, 0]), .8 * .7 * .6 / torch.e, places=6)
        self.assertEqual(float(support[0, 1]), 0.)


if __name__ == '__main__':
    unittest.main()
