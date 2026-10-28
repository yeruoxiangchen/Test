import unittest

import torch

from ...tests.common import synthetic_sample
from ..ray_geometry import CameraBundle, make_patch_rays, candidate_depths
from ..layer_predictor import GeometryConfig, RayLayerPredictor
from ..losses import layer_loss, multiview_point_loss
from ..dataset import select_views
from ..contracts import adapt_v1


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.sample = synthetic_sample()

    def test_projection_roundtrip_and_cv_gl_equivalence(self):
        cameras = CameraBundle.from_sample(self.sample)
        rays = make_patch_rays(cameras, (4, 4))
        depths = torch.full((2, 16, 1), 2.)
        points = rays.points(depths)
        for view in range(2):
            projection = cameras.select(slice(view, view + 1)).project(points[view, :, 0], 4)
            grid = projection['image_grid'][0]
            pixels = (grid + 1) * 63 / 2
            self.assertTrue(torch.allclose(pixels, rays.pixels, atol=1.e-5))
            self.assertTrue(torch.allclose(projection['camera_depth'][0], depths[view, :, 0]))
        flip = torch.diag(torch.tensor([1., -1., -1., 1.]))
        k = cameras.intrinsics.clone(); k[:, 1, 1] *= -1
        gl = CameraBundle.from_tensors(k, flip @ cameras.w2c, 'w2c', -1., cameras.image_size)
        gl_rays = make_patch_rays(gl, (4, 4))
        self.assertTrue(torch.allclose(gl_rays.axial_directions, rays.axial_directions))
        self.assertTrue(torch.allclose(gl_rays.origins, rays.origins))

    def test_candidates_and_ordered_layers_have_gradients(self):
        config = GeometryConfig(input_channels=8, hidden_channels=16, evidence_channels=4, candidates=4, query_chunk=5)
        model = RayLayerPredictor(config)
        sample = self.sample
        cameras = CameraBundle.from_sample(sample)
        rays = make_patch_rays(cameras, (4, 4))
        depths, hits = candidate_depths(rays, cameras, 4)
        self.assertTrue(torch.isfinite(depths).all())
        self.assertTrue((depths[..., 1:] > depths[..., :-1]).all())
        prediction, _ = model.predict(raw_patch_features=sample['raw_visual_patch_features'],
            masked_patch_features=sample['masked_visual_patch_features'], visible_target_mask=sample['visible_target_mask'],
            intrinsics=cameras.intrinsics, extrinsics=cameras.w2c, extrinsics_type='w2c', camera_forward_sign=1., image_size=(64, 64))
        loss, _ = layer_loss(prediction, sample)
        consistency, _ = multiview_point_loss(prediction, sample, cameras)
        (loss + consistency).backward()
        self.assertGreater(float(model.multiview.raw_projection.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.ray_encoder[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_camera_math_stays_fp32_under_autocast(self):
        reference = CameraBundle.from_sample(self.sample)
        reference_rays = make_patch_rays(reference, (4, 4))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cameras = CameraBundle.from_sample(self.sample)
            rays = make_patch_rays(cameras, (4, 4))
            projection = cameras.project(torch.zeros((1, 3)), 4)
        self.assertEqual(rays.axial_directions.dtype, torch.float32)
        self.assertEqual(projection['camera_depth'].dtype, torch.float32)
        self.assertTrue(torch.equal(reference_rays.axial_directions, rays.axial_directions))

    def test_masked_supervision_and_single_view_null(self):
        sample = select_views(adapt_v1(synthetic_sample(channels=1024)), torch.tensor([0]))
        self.assertEqual(len(sample['slat_condition']['cond']), 1)
        self.assertEqual(sample['masked_visual_patch_features'].shape[0], 1)
        sample['masked_visual_patch_features'] = sample['masked_visual_patch_features'][..., :8]
        sample['raw_visual_patch_features'] = sample['raw_visual_patch_features'][..., :8]
        c = GeometryConfig(input_channels=8, hidden_channels=16, evidence_channels=4, candidates=4)
        model = RayLayerPredictor(c)
        cameras = CameraBundle.from_sample(sample)
        prediction, aux = model.predict(raw_patch_features=sample['raw_visual_patch_features'],
            masked_patch_features=sample['masked_visual_patch_features'], visible_target_mask=sample['visible_target_mask'],
            intrinsics=cameras.intrinsics, extrinsics=cameras.w2c, extrinsics_type='w2c', camera_forward_sign=1., image_size=(64, 64))
        self.assertEqual(float(aux['support'].sum()), 0.)
        sample['supervision_valid_mask'] = torch.zeros_like(sample['ray_stop_index_gt'])
        loss, _ = layer_loss(prediction, sample)
        self.assertEqual(float(loss), 0.)


if __name__ == '__main__':
    unittest.main()
