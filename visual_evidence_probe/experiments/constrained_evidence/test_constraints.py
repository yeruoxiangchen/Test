import unittest
import copy
import numpy as np
import torch
from .anchors import closest_points,surface_terms
from .reliability import build as reliable_weights
from .trust import restore


class SurfaceTests(unittest.TestCase):
    def test_face_edge_and_vertex_closest_points(self):
        tri=torch.tensor([[0.,0,0],[1,0,0],[0,1,0]],dtype=torch.float64)
        points=torch.tensor([[.2,.3,1],[2,0,0],[-1,-1,0]],dtype=torch.float64)
        result=closest_points(points,tri)
        expected=torch.tensor([[.2,.3,0],[1,0,0],[0,0,0]],dtype=torch.float64)
        torch.testing.assert_close(result,expected)

    def test_degenerate_triangle_is_finite(self):
        result=closest_points(torch.tensor([0.,1,0]),torch.zeros(3,3))
        self.assertTrue(bool(torch.isfinite(result).all()));torch.testing.assert_close(result,torch.zeros(3))

    def test_actual_surface_loss_is_zero_on_coarse_triangle_without_nearby_vertices(self):
        v=torch.tensor([[0.,0,0],[1,0,0],[0,1,0]],requires_grad=True);f=torch.tensor([[0,1,2]])
        a=dict(points=torch.tensor([[.2,.3,0]]),normals=torch.tensor([[0.,0,1]]))
        result=surface_terms(v,f,a,neighbors=1,scale=1.)
        self.assertLess(float(result['loss']),1.e-12)
        self.assertGreater(float(torch.cdist(a['points'],v).min()),.3)

    def test_surface_gradient_restores_translated_triangle(self):
        v=torch.tensor([[0.,0,.1],[1,0,.1],[0,1,.1]],requires_grad=True);f=torch.tensor([[0,1,2]])
        a=dict(points=torch.tensor([[.2,.3,0]]),normals=torch.tensor([[0.,0,1]]))
        result=surface_terms(v,f,a,neighbors=1,scale=1.);result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(v.grad).all()));self.assertGreater(float(v.grad[:,2].sum()),.19)
        with torch.no_grad():updated=v-.1*v.grad
        self.assertLess(float(surface_terms(updated,f,a,neighbors=1,scale=1.)['loss']),float(result['loss']))


class ReliabilityTests(unittest.TestCase):
    def test_heldout_is_excluded_and_valid_ncc_depths_keep_unit_weight(self):
        depth=torch.ones(3,16,16)*2;valid=torch.ones_like(depth,dtype=torch.bool);valid[2]=False
        target=dict(depth=depth,depth_valid=valid,raw_teacher_depth=depth.clone())
        sparse=dict(depth=depth.clone(),depth_valid=torch.zeros_like(valid));sparse['depth_valid'][0,8,8]=True
        baseline=dict(depth=depth.clone(),mask=torch.ones_like(valid))
        k=torch.tensor([[500.,0,258.5],[0,500.,258.5],[0,0,1.]])[None].repeat(3,1,1)
        w=torch.eye(4)[None].repeat(3,1,1);w[1,0,3]=.1
        gray=torch.zeros_like(depth);eroded=torch.ones_like(valid)
        weights,receipt=reliable_weights(target,sparse,baseline,gray,eroded,k,w,[0,1])
        self.assertEqual(float(weights[0,8,8]),1.)
        self.assertFalse(bool(weights[2].any()))
        self.assertFalse(receipt['heldout_rgb_used']);self.assertFalse(receipt['uses_gt'])
        self.assertAlmostEqual(float(weights[1,8,8]),.1,places=6)


class TrustTests(unittest.TestCase):
    def test_repeated_rollback_restores_moments_without_mutating_backup(self):
        delta=torch.nn.Parameter(torch.ones(2));optimizer=torch.optim.Adam([delta],lr=.1)
        delta.grad=torch.ones(2);optimizer.step()
        saved_delta=delta.detach().clone();saved_state=copy.deepcopy(optimizer.state_dict())
        expected=saved_state['state'][0]['exp_avg'].clone();expected_step=saved_state['state'][0]['step'].clone()
        restore(delta,optimizer,saved_delta,saved_state,.001)
        delta.grad=torch.ones(2)*3;optimizer.step()
        torch.testing.assert_close(saved_state['state'][0]['exp_avg'],expected)
        torch.testing.assert_close(saved_state['state'][0]['step'],expected_step)
        restore(delta,optimizer,saved_delta,saved_state,.001)
        torch.testing.assert_close(delta,saved_delta)
        torch.testing.assert_close(optimizer.state[delta]['exp_avg'],expected)
        torch.testing.assert_close(optimizer.state[delta]['step'],expected_step)


if __name__=='__main__':unittest.main()
