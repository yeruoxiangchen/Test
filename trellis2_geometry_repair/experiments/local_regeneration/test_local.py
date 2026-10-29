import unittest
from types import SimpleNamespace
import numpy as np
import torch

from .sampling import masked_euler,forward_noise,schedule,join_original,restore_exterior
from .fields import compose,keys
from .evidence import connected_core,footprint_fraction
from .generation import compose_support
from .selection import regional_targets


class FakeEuler:
    sigma_min=1.e-5
    def __init__(self):self.inputs=[]
    def sample_once(self,model,value,t,previous,cond=None,**kwargs):
        self.inputs.append((t,value.clone()));return SimpleNamespace(pred_x_prev=value-(t-previous)*model(value,t))


class LocalTests(unittest.TestCase):
    def test_full_noise_schedule_matches_standard_rescaling(self):
        t=schedule(1.,12,3.);self.assertEqual(t[0],1.);self.assertEqual(t[-1],0.)
        self.assertTrue(all(a>b for a,b in zip(t[:-1],t[1:])))
        np.testing.assert_allclose(t,3*np.linspace(1,0,13)/(1+2*np.linspace(1,0,13)))

    def test_partial_sampling_starts_at_requested_time(self):
        s=FakeEuler();clean=torch.ones(3,2);noise=torch.zeros_like(clean)
        masked_euler(s,lambda x,t:torch.zeros_like(x),clean,torch.ones_like(clean,dtype=torch.bool),noise,.35,3,3.,None)
        self.assertEqual(s.inputs[0][0],.35)
        self.assertTrue(torch.equal(s.inputs[0][1],forward_noise(clean,noise,.35)))

    def test_no_region_is_exact_identity_without_model_calls(self):
        s=FakeEuler();clean=torch.randn(3,2)
        value=masked_euler(s,None,clean,torch.zeros(3,1,dtype=torch.bool),torch.randn_like(clean),1.,3,3.,None)
        self.assertTrue(torch.equal(value,clean));self.assertEqual(len(s.inputs),0)

    def test_outside_follows_reference_noise_and_ends_exact(self):
        s=FakeEuler();clean=torch.arange(6).reshape(3,2).float();noise=torch.ones_like(clean)
        active=torch.tensor([[True],[False],[False]])
        result=masked_euler(s,lambda x,t:torch.ones_like(x),clean,active,noise,.75,5,3.,None)
        for t,value in s.inputs:self.assertTrue(torch.equal(value[1:],forward_noise(clean,noise,t)[1:]))
        self.assertTrue(torch.equal(result[1:],clean[1:]))

    def test_full_mask_has_identical_euler_updates(self):
        clean=torch.randn(3,2);noise=torch.randn_like(clean);s=FakeEuler()
        model=lambda x,t:x*.2+t
        result=masked_euler(s,model,clean,torch.ones(3,1,dtype=torch.bool),noise,1.,5,3.,None)
        reference=noise.clone();times=schedule(1.,5,3.)
        for t,previous in zip(times[:-1],times[1:]):reference=reference-(t-previous)*model(reference,t)
        self.assertTrue(torch.equal(result,reference))

    def test_coordinate_join_handles_new_and_reordered_cells(self):
        old=torch.tensor([[0,0,0,0],[0,1,2,3]])
        new=torch.tensor([[0,1,2,3],[0,2,3,4],[0,0,0,0]])
        index,present=join_original(old,new)
        self.assertEqual(present.tolist(),[True,False,True]);self.assertEqual(index[present].tolist(),[1,0])

    def test_restoration_uses_denormalized_original_bits(self):
        old=torch.tensor([[0,0,0,0],[0,1,2,3]])
        features=torch.tensor([[1.234567,.5],[3.345678,-.5]])
        new=old.flip(0);value=torch.zeros_like(features)
        result=restore_exterior(value,features,old,new,torch.tensor([True,False]))
        self.assertTrue(torch.equal(result[1],features[0]))

    def test_empty_field_region_preserves_storage_order_and_values(self):
        base=dict(coords=torch.tensor([[400,1,2],[1,2,3]]),raw=torch.randn(2,7))
        other=dict(coords=torch.tensor([[200,1,2]]),raw=torch.randn(1,7))
        value,receipt=compose(base,other,torch.zeros(32,32,32))
        self.assertTrue(torch.equal(value['coords'],base['coords']));self.assertTrue(torch.equal(value['raw'],base['raw']))
        self.assertTrue(receipt['exterior_fields_exact'])

    def test_field_composition_replaces_core_but_freezes_exterior(self):
        base=dict(coords=torch.tensor([[8,8,8],[400,400,400]]),raw=torch.zeros(2,7))
        other=dict(coords=torch.tensor([[9,8,8],[400,400,400]]),raw=torch.ones(2,7))
        alpha=torch.zeros(32,32,32);alpha[0,0,0]=1.
        result,receipt=compose(base,other,alpha)
        self.assertTrue(torch.equal(result['coords'],torch.tensor([[9,8,8],[400,400,400]])))
        self.assertTrue(torch.equal(result['raw'][1],base['raw'][1]));self.assertTrue(receipt['exterior_fields_exact'])

    def test_field_buffer_blends_existing_cells_without_duplicates(self):
        c=torch.tensor([[16,16,16],[400,400,400]])
        alpha=torch.zeros(32,32,32);alpha[1,1,1]=.5
        result,_=compose(dict(coords=c,raw=torch.zeros(2,7)),dict(coords=c,raw=torch.ones(2,7)),alpha)
        self.assertTrue(torch.equal(result['raw'][0],torch.full((7,),.5)))
        self.assertEqual(len(torch.unique(keys(result['coords']))),2)

    def test_connected_region_does_not_merge_distant_errors(self):
        weights=np.zeros((32,32,32));weights[2,2,2]=10;weights[25,25,25]=1
        core=connected_core(weights,1024)
        self.assertTrue(core[2,2,2]);self.assertFalse(core[25,25,25])

    def test_region_budget_remains_bounded(self):
        weights=np.ones((32,32,32));core=connected_core(weights,128)
        self.assertLessEqual(int(core.sum()),128)

    def test_boundary_voxel_uses_foreground_footprint_instead_of_center(self):
        mask=torch.zeros(8,8,dtype=torch.bool);mask[2:6,4:7]=True
        corners=torch.tensor([[[2.,2.],[4.,2.],[2.,5.],[4.,5.]]])
        self.assertFalse(bool(mask[3,3]))
        fraction=footprint_fraction(mask,corners,image_size=8)
        self.assertAlmostEqual(float(fraction),1/3,places=6)

    def test_footprint_rejects_background_and_outside_image(self):
        mask=torch.zeros(8,8,dtype=torch.bool);mask[0,0]=True
        corners=torch.tensor([[[3.,3.],[5.,5.]], [[-10.,-10.],[-8.,-8.]]])
        self.assertTrue(torch.equal(footprint_fraction(mask,corners,image_size=8),torch.zeros(2)))

    def test_verified_support_anchors_survive_native_threshold_and_preserve_outside(self):
        old=torch.tensor([[0,1,1,1],[0,8,8,8]],dtype=torch.int32)
        proposed=torch.tensor([[0,2,2,2],[0,20,20,20]],dtype=torch.int32)
        core=torch.zeros(32,32,32,dtype=torch.bool);core[:4,:4,:4]=True
        hull=core.clone();positive=torch.zeros_like(core);positive[3,3,3]=True
        coords,receipt=compose_support(old,proposed,core,hull,positive,anchor=True)
        self.assertEqual(coords.tolist(),[[0,2,2,2],[0,3,3,3],[0,8,8,8]])
        self.assertEqual(receipt['covered_after'],receipt['requested']);self.assertEqual(receipt['requested'],1)
        self.assertTrue(receipt['outside_support_exact'])

    def test_native_support_mode_does_not_add_anchors_without_request(self):
        old=torch.tensor([[0,8,8,8]],dtype=torch.int32);proposed=torch.tensor([[0,1,1,1]],dtype=torch.int32)
        core=torch.zeros(32,32,32,dtype=torch.bool);core[:4,:4,:4]=True
        positive=torch.zeros_like(core);positive[2,2,2]=True
        coords,receipt=compose_support(old,proposed,core,core,positive,anchor=False)
        self.assertEqual(coords.tolist(),[[0,1,1,1],[0,8,8,8]]);self.assertEqual(receipt['covered_after'],0)

    def test_local_scoring_keeps_target_or_baseline_inside_fixed_envelope(self):
        # Rays point along +z from cameras at three different x coordinates.
        depth=torch.full((3,1,1),.1);k=torch.eye(3).repeat(3,1,1);w=torch.eye(4).repeat(3,1,1)
        k[:,0,2]=258.5;k[:,1,2]=258.5
        w[:,0,3]=torch.tensor([0.,-1.,-1.])
        trusted=dict(depth=depth,valid=torch.ones_like(depth,dtype=torch.bool))
        xyz=torch.tensor([[[[1.,0.,.1]]],[[[0.,0.,.1]]],[[[1.,0.,.1]]]])
        base=dict(xyz=xyz,mask=torch.ones_like(depth,dtype=torch.bool))
        region=torch.zeros(32,32,32,dtype=torch.bool);region[16,16,19]=True
        result,receipt=regional_targets(trusted,base,torch.eye(4),region,k,w)
        self.assertEqual(result['valid'].flatten().tolist(),[True,True,False])
        self.assertEqual(receipt['local_verified_pixels'],2)
        self.assertTrue(trusted['valid'].all())


if __name__=='__main__':unittest.main()
