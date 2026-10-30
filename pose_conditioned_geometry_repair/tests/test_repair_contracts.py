import unittest
import numpy as np
import torch
from pose_conditioned_geometry_repair.corruptions import (
    corrupt_occupancy,corrupt_shape,roi_mask,remap_reference,ss_edit_mask)
from pose_conditioned_geometry_repair.model import DamagedStateCondition,conditional_sample
from pose_conditioned_geometry_repair.flow import diffuse,target_velocity,weighted_flow_loss,bridge_state,sample_bridge
from pose_conditioned_geometry_repair.prepare import voxel_projection_radius,choose_formal_uids


class RepairContracts(unittest.TestCase):
    def test_formal_selection_preserves_disjointness_and_uses_unseen_reserves(self):
        train,dev=choose_formal_uids(['t0','t1','t2','t3','t4'],['v0'],{'t3'},train_count=2,dev_count=2)
        self.assertEqual(train,['t0','t1'])
        self.assertEqual(dev,['v0','t2'])
        self.assertFalse(set(train)&set(dev))
        with self.assertRaises(ValueError):
            choose_formal_uids(['t0'],[],set(),train_count=2,dev_count=1)

    def test_voxel_projection_tolerance_bounds_corner_displacements(self):
        center=np.array([[.3,-.2,1.4]])
        k=np.array([[640.,0.,255.5],[0.,640.,255.5],[0.,0.,1.]])
        radius=voxel_projection_radius(center,k)
        signs=np.array(np.meshgrid([-1.,1.],[-1.,1.],[-1.,1.])).reshape(3,-1).T
        corners=center+signs/(2*64)
        def project(x):
            return x[:,:2]/x[:,2:3]*640+255.5
        self.assertLessEqual(float(np.abs(project(corners)-project(center)).max()),radius)

    def test_zero_residual_bridge_is_exactly_no_op(self):
        class ZeroModel:
            def _pose_condition(self,*args):
                return None
            def stock_prediction(self,x,*args):
                return torch.zeros_like(x)
            def adapted_prediction(self,x,*args,**kwargs):
                return torch.zeros_like(x),{}
        bad=torch.randn(1,8,4,4,4)
        pair={'bad':bad,'mask':torch.ones(1,1,4,4,4,dtype=torch.bool),
              'present':torch.ones(1,1,4,4,4,dtype=torch.bool)}
        sample={'dino_features':torch.zeros(1,1029,1024)}
        for seed in [42,43]:
            self.assertTrue(torch.equal(sample_bridge(ZeroModel(),sample,pair,seed=seed),bad))

    def test_bridge_input_matches_damaged_start_and_clean_endpoint(self):
        clean=torch.randn(7,32)
        bad=torch.randn_like(clean)
        noise=torch.randn_like(clean)
        self.assertTrue(torch.equal(bridge_state(clean,bad,noise,0),clean))
        self.assertTrue(torch.allclose(bridge_state(clean,bad,noise,1),bad+.1*noise))
        t=.4
        state=bridge_state(clean,bad,noise,t)
        # The supervised residual exactly reconstructs the clean state after
        # removing known feature augmentation. Stock denoising is not used.
        endpoint=state-.1*t*noise-t*(bad-clean)
        self.assertTrue(torch.allclose(endpoint,clean,atol=1e-6))

    def test_condition_whitelist_excludes_all_gt_leaves(self):
        source={'dino_features':torch.zeros(1,1029,1024),'target':torch.ones(1),
                'target_feats':torch.ones(2,32),'pair_pack':{'clean_ss':torch.ones(1)},
                'row':{'source_gt':'must not enter model'}}
        pair={'bad':torch.zeros(1),'mask':torch.ones(1,dtype=torch.bool),
              'present':torch.ones(1,dtype=torch.bool)}
        cond=conditional_sample(source,pair)
        self.assertFalse(any(k in cond for k in ['target','target_feats','pair_pack','row']))
        self.assertIs(cond['repair_reference'],pair['bad'])

    def test_occupancy_locality_and_determinism(self):
        xyz=np.stack(np.meshgrid(np.arange(20,40),np.arange(20,40),np.arange(25,28),indexing='ij'),-1).reshape(-1,3)
        original=np.zeros((64,64,64),bool)
        original[tuple(xyz.T)]=1
        for kind in ['missing','displaced','spurious','identity']:
            with self.subTest(kind=kind):
                bad,bounds,region,changed=corrupt_occupancy(xyz,42,kind)
                self.assertTrue(np.array_equal(bad[~region],original[~region]))
                self.assertTrue(np.array_equal(bad,corrupt_occupancy(xyz,42,kind)[0]))
                self.assertEqual(changed==0,kind=='identity')

    def test_feature_intervention_preserves_exterior(self):
        coords=torch.tensor([[5,5,5],[12,12,12],[14,14,14],[28,28,28]])
        clean=torch.randn(4,32)
        bounds=np.array([[20,20,20],[32,32,32]])
        for kind in ['missing','displaced','spurious','identity']:
            with self.subTest(kind=kind):
                bad,mask,present=corrupt_shape(clean,coords,bounds,42,kind)
                self.assertTrue(torch.equal(bad[~mask],clean[~mask]))
                self.assertTrue(torch.equal(bad,corrupt_shape(clean,coords,bounds,42,kind)[0]))
                if kind=='missing':
                    self.assertFalse(present[mask].any())
                    self.assertEqual(int(bad[mask].count_nonzero()),0)

    def test_sparse_reference_lookup_does_not_fill_from_target(self):
        coords=torch.tensor([[4,3,2],[1,1,1]],dtype=torch.int32)
        feats=torch.randn(2,32)
        q=torch.tensor([[1,1,1],[5,5,5],[4,3,2]],dtype=torch.int32)
        out,present=remap_reference(coords,feats,q)
        self.assertEqual(present.tolist(),[True,False,True])
        self.assertTrue(torch.equal(out[0],feats[1]) and torch.equal(out[2],feats[0]))
        self.assertEqual(int(out[1].count_nonzero()),0)

    def test_full_control_cannot_observe_damaged_state(self):
        model=DamagedStateCondition(32)
        f=torch.randn(5,32,requires_grad=True)
        a=model(f,torch.ones(5),torch.ones(5),torch.randn(5,3),enabled=False)
        self.assertEqual(int(a.count_nonzero()),0)
        a.sum().backward()
        self.assertEqual(int(f.grad.count_nonzero()),0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_repair_condition_uses_present_and_roi(self):
        model=DamagedStateCondition(8)
        f=torch.randn(5,8)
        pos=torch.randn(5,3)
        a=model(f,torch.zeros(5),torch.ones(5),pos)
        b=model(f,torch.ones(5),torch.zeros(5),pos)
        self.assertFalse(torch.equal(a,b))
        a.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_flow_time_and_region_normalization(self):
        x=torch.randn(1,8,16,16,16)
        n=torch.randn_like(x)
        self.assertTrue(torch.equal(diffuse(x,n,1),n))
        self.assertTrue(torch.allclose(diffuse(x,n,0),x+1e-5*n,atol=1e-6))
        v=target_velocity(x,n)
        mask=torch.zeros(1,1,16,16,16,dtype=torch.bool)
        mask[...,2:4,2:4,2:4]=1
        pred=v+mask.expand_as(v).float()*2
        loss,inside,outside=weighted_flow_loss(pred,v,mask,'ss','repair')
        self.assertAlmostEqual(float(inside),4.,places=5)
        self.assertEqual(float(outside),0.)
        self.assertAlmostEqual(float(loss),4.,places=5)

    def test_roi_ss_support_is_nonempty(self):
        mask=ss_edit_mask(np.array([[23,23,23],[40,40,40]]))
        self.assertEqual(tuple(mask.shape),(1,1,16,16,16))
        self.assertTrue(mask.any())
        self.assertTrue(roi_mask(torch.tensor([[15,15,15]]),np.array([[0,0,0],[64,64,64]]),32).all())


if __name__=='__main__':
    unittest.main()
