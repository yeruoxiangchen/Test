"""Regression cases for historical frame/target/camera failures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .assets import C, instance_matrix
from .latents import save_arrays
from .scenes import look_at, check_layer_range, in_polygon
from ...v2.ray_geometry import CameraBundle, make_patch_rays


class Contracts(unittest.TestCase):
    def test_nonuniform_reflection_stays_out_of_camera_rotation(self):
        q = Rotation.from_euler('xyz', [.2, .7, -.3]).as_quat()
        rigid, deformation = instance_matrix({'rotation': q[[3, 0, 1, 2]].tolist(),
            'translation': [3, -2, 4], 'non_uniform_scale': [-.7, 1.2, 2.]})
        normalization = np.diag([.3, .3, .3, 1.])
        normalization[:3, 3] = [.1, .2, -.1]
        canonical_from_world = normalization @ C @ np.linalg.inv(rigid)
        points = np.c_[np.random.default_rng(7).normal(size=(100, 3)), np.ones(100)]
        direct = points @ (normalization @ C @ deformation).T
        via_world = points @ (rigid @ deformation).T @ canonical_from_world.T
        np.testing.assert_allclose(direct, via_world, atol=1.e-12)
        rotation = canonical_from_world[:3, :3] / .3
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        self.assertLess(np.linalg.det(deformation[:3, :3]), 0)

    def test_same_geometry_cannot_acquire_sequence_specific_target(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'target.npz'
            save_arrays(path, coords=np.array([[0, 1, 2]], np.int32), feats=np.ones((1, 8)))
            before = path.read_bytes()
            save_arrays(path, coords=np.array([[0, 1, 2]], np.int32), feats=np.ones((1, 8)))
            self.assertEqual(before, path.read_bytes())
            with self.assertRaises(ValueError):
                save_arrays(path, coords=np.array([[0, 1, 3]], np.int32), feats=np.ones((1, 8)))

    def test_pixel_ray_roundtrip_and_behind_camera(self):
        w = torch.tensor(look_at(np.array([1., -2, 1.]), np.zeros(3), np.array([0., 0, 1.])), dtype=torch.float32)[None]
        k = torch.tensor([[[420., 0, 258.5], [0, 420., 258.5], [0, 0, 1.]]])
        cam = CameraBundle.from_tensors(k, w, 'w2c', 1., (518, 518))
        rays = make_patch_rays(cam, (37, 37))
        xyz = rays.origins[0] + rays.axial_directions[0] * 2
        camera = xyz @ w[0, :3, :3].T + w[0, :3, 3]
        uv = camera @ k[0].T
        np.testing.assert_allclose((uv[:, :2]/uv[:, 2:]).numpy(), rays.pixels.numpy(), atol=1.e-4)
        behind = rays.origins[0] - rays.axial_directions[0]
        self.assertFalse(bool(cam.project(behind, 37)['valid'].any()))

    def test_far_wall_is_not_impossible_layer_supervision(self):
        labels = {'layer_depths_gt': torch.tensor([[[[1., 9.]], [[2., 0.]]]])}
        w = np.eye(4)[None]
        w[0, 2, 3] = 2
        coverage = check_layer_range(labels, w)
        self.assertEqual(coverage, .5)
        self.assertEqual(labels['supervision_valid_mask'].tolist(), [[[True, False]]])

    def test_concave_room_polygon(self):
        polygon = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
        self.assertTrue(in_polygon([.5, 2], polygon))
        self.assertFalse(in_polygon([2, 2], polygon))


if __name__ == '__main__':
    unittest.main()
