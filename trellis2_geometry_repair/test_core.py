import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .geometry import cube_rotations,similarity,error
from .edit import select
from .runtime import FixedBranch,NativePath,decode_native
from .metrics import surface_points


class FakeSparse:
    def __init__(self,feats,coords):self.feats,self.coords=feats,coords
    def replace(self,features):return FakeSparse(features,self.coords)


class Prediction(nn.Module):
    def forward(self,x):return x.replace(x.feats[:,:8])


class FakeDecoder(nn.Module):
    def __init__(self):super().__init__();self.to_subdiv=Prediction()


class CoreTests(unittest.TestCase):
    def test_native_decoder_rejects_default256_and_out_of_range_coords(self):
        features=torch.zeros((1,32));coords=torch.tensor([[0,0,0,0]],dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError,'resolution'):
            decode_native(SimpleNamespace(resolution=256),features,coords)
        with self.assertRaisesRegex(RuntimeError,'coordinates'):
            decode_native(SimpleNamespace(resolution=512),features,torch.tensor([[0,32,0,0]],dtype=torch.int32))

    def test_rotation_search_has24_proper_rotations(self):
        rotations=cube_rotations();self.assertEqual(len(rotations),24)
        for r in rotations:
            np.testing.assert_allclose(r@r.T,np.eye(3));self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_similarity_cannot_introduce_reflections(self):
        r,s,t=similarity(torch.tensor([.3,-.4,.2,.1,.2,.3,3.]),torch.eye(3))
        self.assertAlmostEqual(float(torch.det(r)),1.,places=5);self.assertGreater(float(s),0)

    def test_missing_rays_still_pay_error(self):
        target=dict(depth=torch.ones((1,1,1)),mask=torch.ones((1,1,1),dtype=torch.bool))
        pred=dict(depth=torch.zeros((1,1,1)),mask=torch.zeros((1,1,1),dtype=torch.bool))
        self.assertAlmostEqual(float(error(pred,target)),.1,places=6)

    def test_surface_sampling_is_reproducible(self):
        v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        a=surface_points(v,[[0,1,2]],count=20);b=surface_points(v,[[0,1,2]],count=20)
        np.testing.assert_array_equal(a,b);self.assertTrue((a[:,:2].sum(1)<=1).all())

    def test_repair_gate_abstains_without_target_gain(self):
        structure=dict(large_component_count=1,small_component_area_fraction=0.,total_surface_area=1.)
        base=dict(step=0,loss=1.,local_depth=1.,ious=[.9]*4,photo=.1,structure=structure)
        candidate=dict(base,step=10,loss=.99,queries=100,heldout_pixels=100,
                       outside=dict(mean=0.,p95=0.,lost_fraction=0.))
        self.assertEqual(select([base,candidate])['accepted_step'],0)

    def test_frozen_hooks_survive_checkpoint_backward_and_native_bypass(self):
        decoder=FakeDecoder();coords=torch.arange(4)[:,None].repeat(1,4).int();features=torch.ones((4,32))
        def native(d,z,c):
            d.to_subdiv(FakeSparse(z,c))
            return z[:,:3],torch.tensor([[0,1,2]])
        with patch('trellis2_geometry_repair.runtime.decode_native',side_effect=native):
            branch=FixedBranch(decoder,features,coords)
        try:
            x=(-features).requires_grad_(True)
            def forward(value):
                mask=decoder.to_subdiv(FakeSparse(value,coords)).feats[:,0:1]>0
                return value*mask
            checkpoint(forward,x,use_reentrant=False).sum().backward()
            self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))
            with branch.native_mode():
                self.assertTrue(bool((decoder.to_subdiv(FakeSparse(x,coords)).feats<0).all()))
        finally:branch.close()
        self.assertEqual(len(decoder.to_subdiv._forward_hooks),0)

    def test_edit_path_redecodes_native_topology_each_time(self):
        features=torch.ones((4,32));coords=torch.arange(4)[:,None].repeat(1,4).int()
        def native(decoder,value,positions):
            faces=torch.tensor([[0,1,2]]) if float(value.detach().mean())>0 else torch.tensor([[1,2,3]])
            return value[:,:3],faces
        with patch('trellis2_geometry_repair.runtime.decode_native',side_effect=native):
            path=NativePath(FakeDecoder(),features,coords)
            x=(-features).requires_grad_(True);vertices,faces=path(x)
            self.assertFalse(torch.equal(faces,path.faces))
            vertices.sum().backward();self.assertTrue(bool((x.grad[:,:3]==1).all()))


if __name__=='__main__':unittest.main()
