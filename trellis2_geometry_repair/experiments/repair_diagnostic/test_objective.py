import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .objective import robust_depth,terms
from .deformation import Graph
from .runner import structure,initial_surface_parity,select
from .data import seal_inputs,evaluation_points
from .support_audit import cell_union_distance


def image(depth,hit=True):
    value=torch.as_tensor(depth,dtype=torch.float32).reshape(1,1,-1)
    mask=torch.full_like(value,hit,dtype=torch.bool)
    return dict(depth=value,mask=mask,depth_valid=mask,softmask=mask.float(),
                normal=torch.zeros((*value.shape,3)),normal_valid=torch.zeros_like(mask))


class LossTests(unittest.TestCase):
    def test_large_error_retains_gradient_while_old_cap_saturates(self):
        pred=image([.8]);pred['depth'].requires_grad_(True);target=image([1.]);region=torch.ones((1,1,1))
        old=(pred['depth']-target['depth']).abs().clamp_max(.1).mean()
        old_gradient=torch.autograd.grad(old,pred['depth'])[0]
        value=robust_depth(pred,target,image([.8]),region,[0]);value.backward()
        self.assertEqual(float(old_gradient),0.);self.assertLess(float(pred['depth'].grad),-1.)

    def test_matching_depth_has_zero_loss_and_gradient(self):
        pred=image([1.]);pred['depth'].requires_grad_(True)
        value=robust_depth(pred,image([1.]),image([1.]),torch.ones((1,1,1)),[0]);value.backward()
        self.assertAlmostEqual(float(value),0.,places=7);self.assertEqual(float(pred['depth'].grad),0.)

    def test_deleting_a_large_error_ray_does_not_reduce_loss(self):
        target=image([1.]);base=image([.6]);region=torch.ones((1,1,1))
        initial=robust_depth(base,target,base,region,[0])
        deleted=robust_depth(image([0.],False),target,base,region,[0])
        self.assertGreater(float(deleted),float(initial))

    def test_disappearing_rays_cannot_shrink_denominator(self):
        target=image([1.,1.]);base=image([.8,.8]);pred=image([.8,0.]);pred['mask'][...,1]=False
        initial=robust_depth(base,target,base,torch.ones((1,1,2)),[0])
        after=robust_depth(pred,target,base,torch.ones((1,1,2)),[0])
        self.assertGreater(float(after),float(initial))

    def test_heldout_depth_is_not_part_of_training_loss(self):
        base=image([.8]);target=image([1.]);pred=image([.8]);region=torch.ones((2,1,1))
        for record in (base,target,pred):
            for key,value in record.items():record[key]=value.repeat(2,*([1]*(value.ndim-1)))
        first=robust_depth(pred,target,base,region,[0]);pred['depth'][1]=100.
        self.assertEqual(float(first),float(robust_depth(pred,target,base,region,[0])))

    def test_full_objective_preserves_zero_protection_at_baseline(self):
        base=image([.8,.8]);target=image([1.,1.]);region=torch.tensor([[[1.,0.]]])
        value=terms(base,target,base,region,[0])
        self.assertEqual(float(value['protect']),0.)
        self.assertAlmostEqual(float(value['loss']),float(value['local']),places=6)

    def test_graph_leaves_exterior_vertices_exact_and_has_gradients(self):
        v=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[1.,0.,0.],[1.1,0.,0.],[1.,.1,0.]])
        f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
        render=dict(face_id=torch.zeros((1,1,1),dtype=torch.long),mask=torch.ones((1,1,1),dtype=torch.bool))
        graph=Graph(v,f,render,torch.ones((1,1,1)),[0])
        with torch.no_grad():graph.delta.fill_(.2)
        result=graph();self.assertTrue(torch.equal(result[3:],v[3:]));self.assertFalse(torch.equal(result[:3],v[:3]))
        result[:3].sum().backward();self.assertGreater(float(graph.delta.grad.norm()),0.)

    def test_fast_connectivity_matches_reference_including_duplicate_faces(self):
        from visual_evidence_probe.experiments.dense_evidence.gates import component_summary
        v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,0.,0.],[2.001,0.,0.],[2.,.001,0.]])
        f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32).repeat(256,1)
        fast=structure(v,f);reference=component_summary(v.numpy(),f.numpy())
        for key in ('component_count','large_component_count'):self.assertEqual(fast[key],reference[key])
        self.assertAlmostEqual(fast['total_surface_area'],reference['total_surface_area'],places=5)
        self.assertAlmostEqual(fast['small_component_area_fraction'],reference['small_component_area_fraction'],places=7)

    def test_oracle_loading_requires_both_input_decisions(self):
        with tempfile.TemporaryDirectory() as path:
            with patch('trellis2_geometry_repair.experiments.repair_diagnostic.data.case',return_value=Path(path)):
                with self.assertRaisesRegex(RuntimeError,'both input-only'):seal_inputs(5,42)

    def test_surface_parity_accepts_obj_vertex_reindexing_but_rejects_deformation(self):
        native=torch.tensor([[9.,9.,9.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=torch.tensor([[1,2,3]])
        serialized=native[[3,1,2]].clone();serialized_faces=torch.tensor([[1,2,0]])
        self.assertEqual(initial_surface_parity(native,faces,serialized,serialized_faces),0.)
        serialized[0,2]=.01
        with self.assertRaisesRegex(RuntimeError,'triangle surface differs'):
            initial_surface_parity(native,faces,serialized,serialized_faces)

    def test_gt_surface_sampling_keeps_float64_before_weighted_face_choice(self):
        import trimesh
        # Valid float64 triangles collapse if prematurely cast near0.25 to F32.
        vertices=np.array([[.25,.25,0.],[.25+1.e-9,.25,0.],[.25,.25+1.e-9,0.]],dtype=np.float64)
        mesh=trimesh.Trimesh(vertices=vertices,faces=[[0,1,2]],process=False)
        with patch('trellis2_geometry_repair.experiments.repair_diagnostic.data.read',return_value=dict(mesh='fixture',transform=np.eye(4).tolist())):
            with patch('trimesh.load',return_value=mesh):points=evaluation_points(5)
        self.assertEqual(points.dtype,np.float64);self.assertTrue(np.isfinite(points).all())
        self.assertGreater(float(np.ptp(points[:,0])),0.)

    def gate_rows(self,local_lost=0.,outside=.0):
        base=dict(step=0,loss=1.,local_depth=1.,ious=[.9]*4,photo=None,
                  structure=dict(total_surface_area=1.,large_component_count=1,small_component_area_fraction=0.))
        candidate=dict(base,step=40,loss=.5,local_depth=.5,heldout_pixels=32,
                  depth_audit=dict(queries=32,lost_fraction=local_lost),
                  outside=dict(mean=outside,p95=outside,lost_fraction=0.))
        return [base,candidate]

    def test_large_target_gain_cannot_bypass_exterior_damage_gate(self):
        result=select(self.gate_rows(outside=.01))
        self.assertEqual(result['accepted_step'],0);self.assertFalse(result['checks'][0]['checks']['outside_safe'])

    def test_large_target_gain_cannot_bypass_local_support_loss_gate(self):
        result=select(self.gate_rows(local_lost=.02))
        self.assertEqual(result['accepted_step'],0);self.assertFalse(result['checks'][0]['checks']['local_support_safe'])

    def test_support_distance_uses_cell_volume_and_nearest_union(self):
        centers=np.array([[0.,0.,0.],[1.,0.,0.]])
        values=cell_union_distance([[.01,0.,0.],[.02,0.,0.],[1.,0.,0.]],centers)
        np.testing.assert_allclose(values,[0.,.02-1/64,0.],atol=1.e-12)


if __name__=='__main__':unittest.main()
