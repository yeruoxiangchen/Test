"""CPU mathematics checks; opt-in CUDA raster check with EDITOR_TEST_CUDA=1.

Run with python -B -m unittest
visual_evidence_probe.experiments.editor_support.test_objective -v.
The CUDA test also checks perspective-correct camera depth against an analytic
ray/plane intersection; it does not load a decoder or any model checkpoint.
"""

import os
from pathlib import Path
import unittest

import torch

from .objective import (face_normals, freeze_target, geometry_loss, project_clip,
                        raster_tensor, select_edit_views, world_from_depth)


def render_record(depth, mask=None, softmask=None, normal=None):
    if not torch.is_tensor(depth):
        depth = torch.tensor(depth, dtype=torch.float64)
    mask = torch.ones_like(depth, dtype=torch.bool) if mask is None else mask
    result = dict(depth=depth, mask=mask, softmask=mask.to(depth.dtype) if softmask is None else softmask)
    if normal is not None:
        result.update(normal=normal, normal_valid=mask.clone())
    return result


def cuda_raster_diagnostics():
    """Separate exact interior geometry from CUDA's quantized coverage fringe.

    The installed CUDA backend uses a 1/16-pixel coverage grid, then recomputes
    perspective barycentrics at the original positions and clamps b0/b1. An
    included sample can therefore lie just outside the ideal triangle. The
    analytic plane is valid for interior samples; shader-clamped barycentrics
    are the correct expectation for all covered samples. This diagnostic
    checks both rather than relaxing a single depth tolerance.
    """
    from .config import OUT, digest, now, setup, write
    setup(os.environ.get('EDITOR_TEST_GPU', '3'))
    import nvdiffrast.torch as dr

    device = torch.device('cuda:0')
    context = dr.RasterizeCudaContext(device=device)
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    k = torch.tensor([[[450., 0., 258.5], [0., 450., 258.5], [0., 0., 1.]],
                      [[430., 0., 257.], [0., 440., 259.], [0., 0., 1.]]], device=device)
    w = torch.eye(4, device=device).repeat(2, 1, 1)
    angle = torch.tensor(.12, device=device)
    c, s = angle.cos(), angle.sin()
    w[1, :3, :3] = torch.stack((c, c * 0, s, c * 0, c * 0 + 1, c * 0, -s, c * 0, c)).reshape(3, 3)
    w[1, :3, 3] = torch.tensor([-.1, .05, .2], device=device)
    result = dict(time=now(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  cuda_coverage_subpixel_bits=4, resolutions=[],
                  backend_explanation='coverage snaps to 1/16 pixel; shader recomputes original perspective barycentrics and clamps b0/b1',
                  objective_revision=dict(coverage='ReLU gives zero derivative at a matched silhouette',
                    normal='squared normalized-vector differences give exactly zero self-match loss/gradient; mathematically equal to 1-cosine',
                    cpu_self_match_verified_dtypes=['float32', 'float64']),
                  sources={name: digest(Path(__file__).with_name(name))
                           for name in ('objective.py', 'test_objective.py')},
                  model_loaded=False)
    for side in (32, 128, 256):
        vertices = torch.tensor([[-.65, -.6, 2.], [.7, -.6, 2.5], [0., .7, 2.2]],
                                device=device, dtype=torch.float32, requires_grad=True)
        pred = raster_tensor(vertices, faces, k, w, context, side=side)
        yy, xx = torch.meshgrid(torch.arange(side, device=device, dtype=torch.float64),
                                torch.arange(side, device=device, dtype=torch.float64), indexing='ij')
        ray_x = (((xx + .5) * 518 / side - .5)[None] - k[:, None, None, 0, 2].double()) / k[:, None, None, 0, 0]
        ray_y = (((yy + .5) * 518 / side - .5)[None] - k[:, None, None, 1, 2].double()) / k[:, None, None, 1, 1]
        ray = torch.stack((ray_x, ray_y, torch.ones_like(ray_x)), dim=-1)
        camera = torch.einsum('vj,bij->bvi', vertices.double(), w[:, :3, :3].double()) + w[:, None, :3, 3].double()
        normal = torch.cross(camera[:, 1] - camera[:, 0], camera[:, 2] - camera[:, 0], dim=-1)
        analytic = (normal * camera[:, 0]).sum(-1)[:, None, None] / (ray * normal[:, None, None]).sum(-1)
        # Independent float64 evaluation of the exact shader barycentrics,
        # using the actual float32 clip vertices passed to nvdiffrast.
        clip = project_clip(vertices, k, w).double()
        dx = clip[:, None, None, :, 0] - (2 * (xx + .5) / side - 1)[None, :, :, None] * clip[:, None, None, :, 3]
        dy = clip[:, None, None, :, 1] - (2 * (yy + .5) / side - 1)[None, :, :, None] * clip[:, None, None, :, 3]
        area = torch.stack((dx[..., 1] * dy[..., 2] - dy[..., 1] * dx[..., 2],
                            dx[..., 2] * dy[..., 0] - dy[..., 2] * dx[..., 0],
                            dx[..., 0] * dy[..., 1] - dy[..., 0] * dx[..., 1]), -1)
        bary = area / area.sum(-1, keepdim=True)
        b0, b1 = bary[..., 0].clamp(0, 1), bary[..., 1].clamp(0, 1)
        shader_bary = torch.stack((b0, b1, 1 - b0 - b1), -1)
        camera_z = (vertices[None] @ w[:, 2, :3, None] + w[:, None, 2, 3, None])[..., 0]
        shader_expected = (shader_bary * camera_z[:, None, None].double()).sum(-1)
        hit = pred['mask']
        interior = hit & (bary.min(-1).values >= 0)
        fringe = hit & ~interior
        core = hit & (bary.min(-1).values > .1)
        # Screen-space affine barycentrics turn an out-of-triangle weight into
        # perpendicular distance from its corresponding ideal edge in pixels.
        affine = bary * clip[:, None, None, :, 3]
        affine = affine / affine.sum(-1, keepdim=True)
        screen = (clip[..., :2] / clip[..., 3:4] + 1) * side / 2
        twice_area = torch.cross(torch.cat((screen[:, 1] - screen[:, 0], screen[:, :1, 0] * 0), -1),
                                 torch.cat((screen[:, 2] - screen[:, 0], screen[:, :1, 0] * 0), -1), dim=-1)[..., 2].abs()
        edges = torch.stack(((screen[:, 2] - screen[:, 1]).norm(dim=-1),
                             (screen[:, 0] - screen[:, 2]).norm(dim=-1),
                             (screen[:, 1] - screen[:, 0]).norm(dim=-1)), -1)
        altitude = twice_area[:, None] / edges
        outside_pixels = (-affine * altitude[:, None, None]).clamp_min(0).max(-1).values
        raster_gradient = torch.autograd.grad(pred['depth'][core].mean(), vertices, retain_graph=True)[0]
        analytic_gradient = torch.autograd.grad(analytic[core].mean(), vertices, retain_graph=True)[0]
        target = freeze_target(pred)
        target['depth'][target['mask']] += .02
        terms = geometry_loss(pred, target)
        gradient = torch.autograd.grad(terms['loss'], vertices)[0]
        depth_error = (pred['depth'].double() - analytic).abs()
        shader_error = (pred['depth'].double() - shader_expected).abs()
        views = []
        for view in range(len(k)):
            views.append(dict(view=view, hit_pixels=int(hit[view].sum()),
                interior_pixels=int(interior[view].sum()), fringe_pixels=int(fringe[view].sum()),
                max_plane_error_all=float(depth_error[view][hit[view]].max()),
                max_plane_error_interior=float(depth_error[view][interior[view]].max()),
                max_plane_error_fringe=float(depth_error[view][fringe[view]].max()) if fringe[view].any() else 0.,
                max_shader_error_all=float(shader_error[view][hit[view]].max()),
                max_outside_distance_pixels=float(outside_pixels[view][fringe[view]].max()) if fringe[view].any() else 0.))
        result['resolutions'].append(dict(side=side, views=views,
            core_pixels=int(core.sum()), gradient_norm=float(gradient.norm()),
            gradient_finite=bool(torch.isfinite(gradient).all()),
            analytic_depth_gradient_max_error=float((raster_gradient - analytic_gradient).abs().max()),
            analytic_depth_gradient_relative_error=float((raster_gradient - analytic_gradient).norm() / analytic_gradient.norm())))
    result['diagnostic_path'] = str(OUT / 'raster_camera_gradient_diagnostic.json')
    write(result['diagnostic_path'], result)
    return result


class ObjectiveMathTest(unittest.TestCase):
    def test_projection_and_backprojection_match_pixel_centers(self):
        dtype = torch.float64
        depth = torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4) / 10 + 1
        k = torch.tensor([[[450., 0., 258.5], [0., 430., 257.], [0., 0., 1.]],
                          [[380., 0., 260.], [0., 410., 255.], [0., 0., 1.]]], dtype=dtype)
        w = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        w[1, :3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=dtype)
        w[:, :3, 3] = torch.tensor([[.1, -.2, 2.], [-.2, .1, 3.]], dtype=dtype)
        world = world_from_depth(depth, k, w)
        for view in range(2):
            clip = project_clip(world[view].reshape(-1, 3), k[view:view+1], w[view:view+1])[0]
            ndc = clip[:, :2] / clip[:, 3:4]
            yy, xx = torch.meshgrid(torch.arange(3, dtype=dtype), torch.arange(4, dtype=dtype), indexing='ij')
            expected = torch.stack((2 * (xx + .5) / 4 - 1, 2 * (yy + .5) / 3 - 1), -1).reshape(-1, 2)
            torch.testing.assert_close(ndc, expected, atol=1.e-12, rtol=1.e-12)
            torch.testing.assert_close(clip[:, 3], depth[view].reshape(-1))
        self.assertTrue(torch.autograd.gradcheck(lambda z: world_from_depth(z, k, w),
                                               (depth.clone().requires_grad_(),)))

    def test_cross_product_normals_have_vertex_derivatives(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., .2]],
                                dtype=torch.float64, requires_grad=True)
        faces = torch.tensor([[0, 1, 2]])
        normal, valid = face_normals(vertices, faces)
        expected = torch.tensor([[0., -.2, 1.]], dtype=vertices.dtype)
        expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
        torch.testing.assert_close(normal, expected)
        self.assertTrue(valid.all())
        self.assertTrue(torch.autograd.gradcheck(lambda value: face_normals(value, faces)[0], (vertices,)))

    def test_local_edit_and_complement_protection_have_expected_gradient(self):
        depth = torch.tensor([[[1.03, 1.01]]], dtype=torch.float64, requires_grad=True)
        target_depth = torch.tensor([[[1., 8.]]], dtype=torch.float64, requires_grad=True)
        pred = render_record(depth)
        target = render_record(target_depth)
        baseline = render_record([[[7., 1.]]])
        region = torch.tensor([[[True, False]]])
        terms = geometry_loss(pred, target, baseline, region,
                              weights=dict(silhouette=0., normal=0., coverage=0., protect=2.))
        self.assertAlmostEqual(float(terms['local_depth']), .6)
        self.assertAlmostEqual(float(terms['protect_depth']), .2)
        self.assertAlmostEqual(float(terms['loss']), 1.)
        terms['loss'].backward()
        torch.testing.assert_close(depth.grad, torch.tensor([[[20., 40.]]], dtype=depth.dtype))
        self.assertIsNone(target_depth.grad)

    def test_disappearing_bad_ray_cannot_reduce_fixed_domain_loss(self):
        target = render_record([[[1., 1.]]])
        before = render_record([[[2., 1.]]])
        hit = torch.tensor([[[False, True]]])
        after = render_record([[[0., 1.]]], hit)
        weights = dict(normal=0., silhouette=0., coverage=.2)
        initial = geometry_loss(before, target, weights=weights)
        removed = geometry_loss(after, target, weights=weights)
        self.assertAlmostEqual(float(initial['local_depth']), 1.)
        self.assertAlmostEqual(float(removed['local_depth']), 1.)
        self.assertGreater(float(removed['loss']), float(initial['loss']))

    def test_sparse_depth_domain_and_separate_normal_protection(self):
        depth = torch.tensor([[[1.02, 17., 1.]]], dtype=torch.float64, requires_grad=True)
        normal = torch.tensor([[[[0., 0., 1.], [0., 0., 1.], [.6, 0., .8]]]],
                              dtype=torch.float64, requires_grad=True)
        pred = render_record(depth, normal=normal)
        target = render_record([[[1., 3., 9.]]])  # No local target normals exist.
        target['depth_valid'] = torch.tensor([[[True, False, False]]])
        baseline = render_record([[[1., 1., 1.]]], normal=torch.tensor(
            [[[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]]]], dtype=depth.dtype))
        region = torch.tensor([[[True, True, False]]])
        terms = geometry_loss(pred, target, baseline, region, weights=dict(
            depth=1., silhouette=0., coverage=0., local_normal=0., protect_normal=1.))
        self.assertAlmostEqual(float(terms['local_depth']), .4)
        self.assertAlmostEqual(float(terms['local_normal']), 0.)
        self.assertAlmostEqual(float(terms['protect_normal']), .2)
        terms['loss'].backward()
        self.assertEqual(float(depth.grad[0, 0, 1]), 0.)
        self.assertGreater(float(normal.grad[0, 0, 2].abs().sum()), 0.)

    def test_identical_reference_including_soft_boundary_is_zero(self):
        record = render_record([[[1., 1.]]], softmask=torch.tensor([[[.25, 1.]]], dtype=torch.float64))
        self.assertEqual(float(geometry_loss(record, freeze_target(record))['loss']), 0.)

    def test_matching_target_has_zero_gradient_for_every_component(self):
        for dtype in (torch.float32, torch.float64):
            depth = torch.tensor([[[1.1, 1.2]]], dtype=dtype, requires_grad=True)
            softmask = torch.tensor([[[.25, .7]]], dtype=dtype, requires_grad=True)
            normal = torch.tensor([[[[.1, -.2, .7], [.3, .1, -.6]]]], dtype=dtype, requires_grad=True)
            pred = render_record(depth, softmask=softmask, normal=normal)
            target = freeze_target(pred)
            terms = geometry_loss(pred, target, target, local_region=torch.tensor([[[1., 0.]]]))
            for name, value in terms.items():
                self.assertEqual(float(value), 0., (dtype, name))
                gradients = torch.autograd.grad(value, (depth, softmask, normal), retain_graph=True, allow_unused=True)
                for gradient in gradients:
                    if gradient is not None:
                        self.assertEqual(float(gradient.abs().max()), 0., (dtype, name))

    def test_training_views_exclude_holdout_and_target_is_immutable(self):
        depth = torch.tensor([[[1.]], [[5.]]], dtype=torch.float64, requires_grad=True)
        target = freeze_target(render_record(torch.ones_like(depth)))
        pred = render_record(depth)
        terms = geometry_loss(pred, target, view_indices=[0])
        self.assertEqual(float(terms['loss']), 0.)
        terms['loss'].backward()
        self.assertEqual(float(depth.grad[1].sum()), 0.)
        original = render_record(torch.ones_like(depth))
        frozen = freeze_target(original)
        original['depth'][0] = 9
        self.assertEqual(float(frozen['depth'][0, 0, 0]), 1.)

    def test_holdout_requires_local_visibility_and_is_not_fixed_view_four(self):
        xyz = torch.zeros((4, 2, 2, 3), dtype=torch.float64)
        xyz[2:] = 10
        record = dict(xyz=xyz, mask=torch.ones((4, 2, 2), dtype=torch.bool))
        k = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1)
        w = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
        w[:, :3, 3] = torch.tensor([[0., 0., 2.], [2., 0., 0.], [0., 2., 0.], [0., 0., -2.]])
        selected = select_edit_views(record, [[0., 0., 0.]], k, w, radius=.1, dilation=0)
        self.assertEqual(selected['status'], 'ok')
        self.assertEqual(selected['visible_views'], [0, 1])
        self.assertIn(selected['heldout_views'][0], [0, 1])
        self.assertEqual(len(selected['train_views']), 1)
        invisible = select_edit_views(record, [[5., 5., 5.]], k, w, radius=.1)
        self.assertEqual(invisible['status'], 'unable')
        self.assertEqual(invisible['heldout_views'], [])

    @unittest.skipUnless(os.environ.get('EDITOR_TEST_CUDA') == '1', 'opt-in CUDA raster check')
    def test_raster_depth_matches_analytic_ray_plane_and_backpropagates(self):
        diagnostic = cuda_raster_diagnostics()
        for row in diagnostic['resolutions']:
            self.assertGreater(row['core_pixels'], 10)
            self.assertTrue(row['gradient_finite'])
            self.assertGreater(row['gradient_norm'], 0.)
            self.assertLess(row['analytic_depth_gradient_max_error'], 1.e-4)
            for view in row['views']:
                self.assertGreater(view['interior_pixels'], 10)
                # Keep the original plane tolerance and add a stricter
                # independently derived shader check on every covered sample.
                self.assertLess(view['max_plane_error_interior'], 3.e-5)
                self.assertLess(view['max_shader_error_all'], 3.e-6)
                self.assertLessEqual(view['max_outside_distance_pixels'], 2 ** .5 / 32 + 1.e-5)


if __name__ == '__main__':
    unittest.main()
