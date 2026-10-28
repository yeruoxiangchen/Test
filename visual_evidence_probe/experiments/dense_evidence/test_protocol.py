import unittest
import numpy as np
import torch

from .camera import remap_depth,backproject,project,consistency,pixel_grid
from .gates import select,observed_decision
from .evidence import augment_depth
from .report import object_summary,cluster


class CameraTests(unittest.TestCase):
    def cameras(self):
        k=torch.tensor([[500.,0,258.5],[0,500.,258.5],[0,0,1.]])[None].repeat(3,1,1)
        w=torch.eye(4)[None].repeat(3,1,1);w[1,0,3]=.08;w[2,0,3]=-.08
        return k,w

    def test_camera_z_remapping_uses_intrinsics(self):
        k,_=self.cameras();pk=k.clone();pk[:,:2]*=504/518
        depth=np.full((3,504,504),2.,np.float32)
        result,_,valid=remap_depth(depth,None,pk,k,32)
        self.assertLess(float((result[valid]-2).abs().max()),1.e-6)

    def test_backprojection_roundtrip(self):
        k,w=self.cameras();d=torch.ones(32,32)*2
        point=backproject(d,k[1],w[1]);uv,z=project(point,k[1],w[1])
        self.assertLess(float((uv-pixel_grid(32,d.device)[...,:2]).abs().max()),1.e-4)
        self.assertLess(float((z-d).abs().max()),1.e-6)

    def test_consistency_retains_correct_plane_and_rejects_conflicting_depth(self):
        k,w=self.cameras();d=torch.ones(3,32,32)*2;mask=torch.ones_like(d,dtype=torch.bool)
        conf=torch.ones_like(d)
        good,_,_=consistency(d,conf,mask,k,w)
        self.assertGreater(float(good[:,4:-4,4:-4].float().mean()),.95)
        broken=d.clone();broken[1:]=3
        bad,_,_=consistency(broken,conf,mask,k,w)
        self.assertEqual(int(bad[0].sum()),0)

    def test_foreground_and_confidence_domains_are_fixed_without_gt(self):
        k,w=self.cameras();d=torch.ones(3,32,32)*2;mask=torch.ones_like(d,dtype=torch.bool);mask[:,:16]=False
        valid,_,receipt=consistency(d,torch.ones_like(d),mask,k,w)
        self.assertFalse(bool(valid[:,:16].any()));self.assertFalse(receipt['uses_gt'])


class GateTests(unittest.TestCase):
    def test_dense_augmentation_retains_sparse_anchors_and_falls_back_exactly(self):
        sparse=torch.tensor([[[0.,2.],[0.,0.]]]);sv=sparse>0
        raw=torch.ones_like(sparse)*3;valid=torch.zeros_like(sv)
        merged,mv=augment_depth(raw,valid,sparse,sv)
        self.assertTrue(torch.equal(mv,sv));self.assertEqual(float(merged[sv][0]),2.)
        valid[0,1,1]=True
        merged,mv=augment_depth(raw,valid,sparse,sv)
        self.assertTrue(bool(mv[0,1,1]));self.assertEqual(float(merged[0,1,1]),3.)
    def state(self,step=0):
        structure=dict(component_count=1,nonmanifold_edge_count=0,large_component_count=1,
            small_component_area_fraction=0.,total_surface_area=1.)
        return dict(step=step,loss=.1 if step==0 else .05,local_depth=.1 if step==0 else .05,
            iou_original=[1.,1.,1.,1.] if step==0 else [.90]*4,
            iou_observed=[.90]*4 if step==0 else [.95]*4,
            validation_cost=.2 if step==0 else .199,
            outside=dict(depth_mae=0.,depth_p95=0.),structure=structure)

    def test_corrected_contour_can_pass_observed_policy_while_legacy_rejects(self):
        base,candidate=self.state(),self.state(20)
        result=select([base,candidate],base['structure'],128,32)
        self.assertEqual(result['legacy_step'],0);self.assertEqual(result['observed_step'],20)

    def test_observed_regression_is_not_hidden_by_preserved_original_shape(self):
        base,candidate=self.state(),self.state(20)
        candidate['iou_observed']=[.88]*4;candidate['iou_original']=[.999]*4
        result=select([base,candidate],base['structure'],128,32)
        self.assertEqual(result['observed_step'],0)

    def test_larger_component_damage_is_rejected_and_tiny_fragment_budget_is_explicit(self):
        base,candidate=self.state(),self.state(20)
        candidate['structure']['large_component_count']=2
        self.assertFalse(observed_decision(candidate,base,base['structure'],128,32)['accepted'])
        candidate['structure']['large_component_count']=1;candidate['structure']['small_component_area_fraction']=.0001
        self.assertTrue(observed_decision(candidate,base,base['structure'],128,32)['accepted'])

    def test_no_evidence_or_no_heldout_support_restores_baseline(self):
        base,candidate=self.state(),self.state(20)
        self.assertEqual(select([base,candidate],base['structure'],0,32)['observed_step'],0)
        self.assertEqual(select([base,candidate],base['structure'],128,0)['observed_step'],0)


class ReportTests(unittest.TestCase):
    def test_seeds_are_averaged_before_objects(self):
        result=object_summary([('a',0.),('a',2.),('b',3.)])
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['objects'],2)

    def test_real_groups_share_one_physical_object_cluster(self):
        a=cluster(dict(kind='real_scene',uid='A'))
        b=cluster(dict(kind='real_scene',uid='B'))
        result=object_summary([(a,1.),(b,2.)])
        self.assertEqual(result['objects'],1)
        self.assertIsNone(result['ci95'])


if __name__=='__main__':unittest.main()
