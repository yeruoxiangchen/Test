import unittest

import numpy as np
import torch

from .policy import choose_source, select_checkpoint
from .refine import project_clip, transform


class InputPolicyTests(unittest.TestCase):
    def test_both_views_must_pass_and_ties_are_stable(self):
        self.assertEqual(choose_source({'a': [.99, .79], 'b': [.85, .85]}, .8), 'b')
        self.assertEqual(choose_source({'b': [.85, .85], 'a': [.85, .85]}, .85), 'a')
        self.assertIsNone(choose_source({'a': [.89, .99], 'b': [.99, .89]}, .9))
        with self.assertRaises(ValueError):
            choose_source({'a': [float('nan'), .9]})

    def test_checkpoint_cannot_sacrifice_a_view(self):
        snapshots = [{'step': 0, 'ious': [.8, .8]}, {'step': 20, 'ious': [.99, .79]},
                     {'step': 40, 'ious': [.82, .82]}]
        self.assertEqual(select_checkpoint(snapshots)['step'], 40)
        self.assertEqual(select_checkpoint(snapshots[:2])['step'], 0)

    def test_proper_similarity_bounds_identity_and_projection(self):
        r, t, s = transform(torch.zeros(7))
        self.assertTrue(torch.equal(r, torch.eye(3)))
        self.assertTrue(torch.equal(t, torch.zeros(3)))
        self.assertEqual(float(s), 1.)
        r, t, s = transform(torch.tensor([20., -10., 5., 20., -20., 30., 20.]))
        self.assertAlmostEqual(float(torch.det(r)), 1., places=5)
        self.assertTrue(bool((t.abs() <= .15).all()))
        self.assertTrue(.75 <= float(s) <= 1.25)
        k = torch.tensor([[[400., 0., 258.5], [0., 400., 258.5], [0., 0., 1.]]])
        w = torch.eye(4)[None]
        vertices = torch.tensor([[.1, -.2, 2.], [-.2, .1, 3.]])
        clip = project_clip(vertices, k, w)[0]
        pixel = (clip[:, :2] / clip[:, 3:] + 1) * 518 / 2 - .5
        expected = vertices[:, :2] / vertices[:, 2:] * 400 + 258.5
        self.assertTrue(torch.allclose(pixel, expected, atol=3.e-5, rtol=0))


if __name__ == '__main__':
    unittest.main()
