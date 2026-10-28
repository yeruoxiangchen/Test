import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import photometric as photo
from .config import PARAMS
from .deform import solve, area_region


class RepairTests(unittest.TestCase):
    def test_patch_brightness_affine(self):
        torch.manual_seed(42)
        a = torch.rand(32, 32)
        cost, _, _ = photo.patch_cost(a, a * .6 + .2)
        self.assertLess(float(cost[3:-3, 3:-3].max()), 1.e-5)

    def test_depth_sweep_sign_and_scale(self):
        # Render a shared, non-periodic world texture on z=2 with shifted cameras.
        side = 128
        k = torch.tensor([[400., 0, 258.5], [0, 400., 258.5], [0, 0, 1.]])
        ws = torch.eye(4).repeat(4, 1, 1)
        ws[:, 0, 3] = torch.tensor([0., -.35, .35, -.6])
        zs = torch.full((side, side), 2.)
        images = []
        for w in ws:
            xyz, _ = photo.world_from_depth(zs, k, w)
            x, y = xyz[..., 0], xyz[..., 1]
            image = .5 + .11 * torch.sin(37*x + 9*y) + .12*torch.cos(21*y-17*x)
            image += .1*torch.sin(9*x*x + 31*y + 12*x)
            images.append(image)
        # True plane z2, initial plane z2.03. Search must move toward the cameras.
        initial = torch.full((4, side, side), 2.03)
        pred = dict(depth=initial, mask=torch.ones_like(initial, dtype=torch.bool))
        eroded = torch.ones_like(initial, dtype=torch.bool)
        # Pair-volume is CPU-compatible; relax only visibility to include this synthetic displacement.
        with patch.dict(PARAMS, visibility_tolerance=.05, min_texture_std=.005,
                        min_ncc_gain=.001, pair_depth_agreement=.02):
            proposals = photo.propose(pred, torch.stack(images), eroded, k.repeat(4, 1, 1), ws)
        selected = proposals['delta'][proposals['valid']]
        self.assertGreater(len(selected), 100)
        self.assertAlmostEqual(float(np.median(selected)), -.03, delta=.006)

    def test_local_solver_preserves_outside_and_cap(self):
        import trimesh
        grid = np.stack(np.meshgrid(np.arange(5), np.arange(5)), -1).reshape(-1, 2) / 4
        vertices = np.c_[grid, np.zeros(25)]
        faces = []
        for y in range(4):
            for x in range(4):
                a = y * 5 + x
                faces += [[a, a+1, a+5], [a+1, a+6, a+5]]
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        selected = np.zeros(len(faces), bool); selected[[10, 11, 12, 13]] = True
        ids = np.flatnonzero(selected)
        c = dict(face_ids=ids, bary=np.full((4, 3), 1/3),
                 displacement=np.tile([0, 0, .02], (4, 1)), confidence=np.ones(4))
        offset, result = solve(mesh, c, selected)
        movable = np.unique(np.asarray(faces)[selected])
        outside = np.ones(25, bool); outside[movable] = False
        self.assertTrue(np.array_equal(offset[outside], np.zeros((outside.sum(), 3))))
        self.assertGreater(offset[:, 2].max(), 0)
        self.assertLessEqual(np.linalg.norm(offset, axis=1).max(), PARAMS['max_displacement'] + 1.e-8)
        self.assertTrue(result['frozen_vertices_exact'])

    def test_region_budget_controls(self):
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=2)
        ids = np.arange(len(mesh.faces)).reshape(20, -1)
        scores = np.arange(ids.size).reshape(ids.shape)
        a, ar = area_region(mesh, [ids], [scores], 'automatic', 42)
        b, br = area_region(mesh, [ids], [scores], 'random', 42)
        self.assertLess(abs(ar['selected_area'] - br['selected_area']), 2 * mesh.area_faces.max())
        self.assertFalse(np.array_equal(a, b))


if __name__ == '__main__':
    unittest.main()
