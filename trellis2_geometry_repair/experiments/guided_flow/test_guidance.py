import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from .integration import clean_correction,integrate
from .objective import log_surface,routing_loss,surface_loss,match,field_protect,route_values,missing_cost,surface_costs,protection_costs,fixed_depth_cost,free_loss
from .data import query_ids,record
from .selection import budget_prefix
from ..local_regeneration.sampling import masked_euler

class FakeSampler:
    sigma_min=1.e-5
    def sample_once(self,model,x,t,previous,cond=None,**kwargs):
        v=x.feats*.1+t
        clean=(1-self.sigma_min)*x.feats-(self.sigma_min+(1-self.sigma_min)*t)*v
        return SimpleNamespace(pred_x_prev=SimpleNamespace(feats=x.feats-(t-previous)*v),pred_x_0=SimpleNamespace(feats=clean))

class GuidanceTests(unittest.TestCase):
    def test_euler_correction_matches_upstream_velocity_conversion(self):
        x=torch.randn(5,3);v=torch.randn_like(x);delta=torch.randn_like(x)*.1
        t=.5;previous=.2;sigma=1.e-5;a=sigma+(1-sigma)*t
        clean=(1-sigma)*x-a*v;corrected=clean+delta
        expected=x-(t-previous)*((1-sigma)*x-corrected)/a
        got=clean_correction(x-(t-previous)*v,clean,corrected,t,previous,sigma)
        torch.testing.assert_close(got,expected)

    def test_last_euler_correction_retains_sigma_min_factor(self):
        a=clean_correction(torch.zeros(1),torch.zeros(1),torch.ones(1),.2,0.,.01)
        self.assertAlmostEqual(float(a),.2/(.01+.99*.2),places=6)

    def test_no_guide_matches_v2_sampler_exactly(self):
        sampler=FakeSampler();clean=torch.randn(4,3);noise=torch.randn_like(clean);active=torch.tensor([[True],[False],[True],[False]])
        wrap=lambda x:SimpleNamespace(feats=x)
        expected=masked_euler(sampler,None,clean,active,noise,1.,12,3.,None,wrap=wrap,unwrap=lambda x:x.feats)
        with patch('trellis2_geometry_repair.runtime.sparse',side_effect=lambda x,c:wrap(x)):
            got=integrate(sampler,None,clean,active,noise,None,dict(_coords=None))
        self.assertTrue(torch.equal(got,expected))

    def test_guide_calls_and_exterior_are_exact(self):
        sampler=FakeSampler();clean=torch.randn(4,3);noise=torch.randn_like(clean);active=torch.tensor([[True],[False],[True],[False]])
        calls=[]
        def guide(z,mask,step,t):
            calls.append(step);self.assertTrue(torch.equal(z[1],clean[1]));return torch.where(mask,z+.1,z)
        with patch('trellis2_geometry_repair.runtime.sparse',side_effect=lambda x,c:SimpleNamespace(feats=x)):
            got=integrate(sampler,None,clean,active,noise,None,dict(_coords=None),guide=guide)
        self.assertEqual(calls,[9,10,11]);self.assertTrue(torch.equal(got[1],clean[1]));self.assertTrue(torch.equal(got[3],clean[3]))

    def test_soft_edge_event_is_stable_with_no_hard_surface(self):
        raw=torch.full((2,7),-80.,requires_grad=True);p=log_surface(raw)
        self.assertTrue(torch.isfinite(p).all());self.assertTrue((raw[:,3:6]<=0).all())
        (-p.mean()).backward();self.assertGreater(float(raw.grad[:,3:6].norm()),0.)
        self.assertTrue((raw.grad[:,3:6]<0).all())

    def test_soft_edge_event_is_probability_of_any_crossing(self):
        raw=torch.zeros(1,7)
        self.assertAlmostEqual(float(log_surface(raw).exp()),.875,places=6)

    def test_absent_child_gets_route_gradient_in_correct_native_channel(self):
        raw=torch.full((1,8),-10.,requires_grad=True)
        # child=(5,6,9): parent=(2,3,4), channel=1+0+4=5.
        point=(torch.tensor([[5.5,6.5,9.5]])/64)-.5
        value,coverage=routing_loss([dict(coords=torch.tensor([[2,3,4]]),raw=raw,grid=32)],point)
        value.backward();self.assertLess(float(raw.grad[0,5]),0.)
        self.assertEqual(int((raw.grad!=0).sum()),1);self.assertEqual(coverage[0]['reachable_parent'],1)

    def test_large_half_precision_routing_reduction_stays_finite(self):
        raw=torch.full((4096,8),100.,dtype=torch.float16,requires_grad=True)
        coords=torch.cartesian_prod(torch.arange(16),torch.arange(16),torch.arange(16))
        value,_=routing_loss([dict(coords=coords,raw=raw,grid=32)],torch.zeros(1,3))
        self.assertTrue(bool(torch.isfinite(value)));value.backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_unreachable_coarse_parent_is_reported_not_fabricated(self):
        raw=torch.zeros(1,8,requires_grad=True);point=torch.tensor([[.4,.4,.4]])
        value,coverage=routing_loss([dict(coords=torch.tensor([[0,0,0]]),raw=raw,grid=32)],point)
        value.backward();self.assertEqual(coverage[0]['reachable_parent'],0);self.assertEqual(float(raw.grad.norm()),0.)

    def test_deleting_parent_cannot_lower_frozen_routing_objective(self):
        points=(torch.tensor([[5.5,6.5,9.5]])/64)-.5
        original=dict(coords=torch.tensor([[2,3,4]]),raw=torch.full((1,8),-20.),grid=32)
        cost,valid=route_values(original,points);frozen=missing_cost(cost,valid)
        before,_=routing_loss([original],points,[frozen])
        gone=dict(coords=torch.empty(0,3,dtype=torch.long),raw=torch.empty(0,8,requires_grad=True),grid=32)
        after,stats=routing_loss([gone],points,[frozen])
        self.assertGreater(float(after),float(before));self.assertEqual(stats[0]['fixed_denominator'],1)

    def test_deleting_all_fine_surface_cells_is_penalized(self):
        points=torch.zeros(1,3);fields=dict(coords=torch.tensor([[255,255,255]]),raw=torch.zeros(1,7))
        cost,valid=surface_costs(fields,points);frozen=missing_cost(cost,valid)
        before,_=surface_loss(fields,points,torch.ones(1),frozen)
        after,_=surface_loss(dict(coords=torch.empty(0,3,dtype=torch.long),raw=torch.empty(0,7)),points,torch.ones(1),frozen)
        self.assertGreater(float(after),float(before))

    def test_deleting_exterior_anchors_is_penalized(self):
        fields=dict(coords=torch.tensor([[255,255,255]]),raw=torch.randn(1,7),subs=[])
        costs,valid=protection_costs(fields,fields);frozen=[missing_cost(c,v) for c,v in zip(costs,valid)]
        before=field_protect(fields,fields,frozen)
        gone=dict(coords=torch.empty(0,3,dtype=torch.long),raw=torch.empty(0,7),subs=[])
        after=field_protect(gone,fields,frozen);self.assertGreater(float(after),float(before))

    def test_depth_query_denominator_is_fixed_when_mesh_hit_disappears(self):
        trusted=dict(depth=torch.ones(1,1,2),valid=torch.ones(1,1,2,dtype=torch.bool))
        pred=dict(depth=torch.tensor([[[.8,.6]]]),mask=torch.ones(1,1,2,dtype=torch.bool))
        missing=(pred['depth']-trusted['depth']).abs()+.1
        before=fixed_depth_cost(pred,trusted,[0],missing)
        after=fixed_depth_cost(dict(depth=pred['depth'],mask=torch.zeros_like(pred['mask'])),trusted,[0],missing)
        self.assertGreater(float(after),float(before))

    def test_free_space_has_fixed_sample_count_and_rewards_removing_false_surface(self):
        coords=torch.tensor([[255,255,255],[256,255,255]]);raw=torch.zeros(2,7,requires_grad=True)
        points=(torch.tensor([[255.5,255.5,255.5],[256.5,255.5,255.5],[260.5,260.5,260.5]])/512)-.5
        value,count=free_loss(dict(coords=coords,raw=raw),points,torch.empty(0,3))
        self.assertAlmostEqual(float(value),float(torch.log(torch.tensor(2.)))*2/3,places=6);self.assertEqual(count,2)

    def test_missing_surface_loss_supplies_field_gradient(self):
        raw=torch.full((3,7),-12.,requires_grad=True);coords=torch.tensor([[255,255,255],[256,255,255],[255,256,255]])
        value,stats=surface_loss(dict(coords=coords,raw=raw),torch.tensor([[0.,0.,0.]]),torch.ones(1))
        value.backward();self.assertGreater(float(raw.grad[:,3:6].norm()),0.);self.assertEqual(stats['queries'],1)

    def test_match_does_not_alias_missing_cell_to_existing_row(self):
        _,present=match(torch.tensor([[2,3,4]]),torch.tensor([[2,3,4],[3,4,5]]))
        self.assertEqual(present.tolist(),[True,False])

    def test_out_of_bounds_query_does_not_alias_packed_spatial_key(self):
        _,present=match(torch.tensor([[0,0,1]]),torch.tensor([[-1,512,1],[0,0,1]]))
        self.assertEqual(present.tolist(),[False,True])

    def test_query_cap_is_frozen_and_view_balanced(self):
        valid=torch.ones(4,16,16,dtype=torch.bool)
        ids=query_ids(valid,[0,2,3],24);self.assertEqual(len(ids),24)
        self.assertEqual(torch.bincount(ids//256,minlength=4).tolist(),[8,0,8,8])
        self.assertTrue(torch.equal(ids,query_ids(valid,[0,2,3],24)))

    def test_field_reference_has_zero_gradient_at_identity(self):
        raw=torch.randn(2,7,requires_grad=True);coords=torch.tensor([[2,3,4],[4,5,6]])
        value=field_protect(dict(coords=coords,raw=raw,subs=[]),dict(coords=coords,raw=raw.detach().clone(),subs=[]))
        value.backward();self.assertLess(float(raw.grad.norm()),1.e-7)

    def test_budget_prefix_is_selected_from_cost_without_quality_or_GT(self):
        rows=[dict(sampling_seconds=2.,score_seconds=1.) for _ in range(8)]
        selected,receipt=budget_prefix(rows,10.)
        self.assertEqual(len(selected),4);self.assertTrue(receipt['sufficient']);self.assertEqual(receipt['prior_seconds'],12.)

    def test_insufficient_budget_bank_is_explicit(self):
        rows=[dict(sampling_seconds=1.,score_seconds=0.) for _ in range(8)]
        _,receipt=budget_prefix(rows,20.);self.assertFalse(receipt['sufficient']);self.assertEqual(receipt['prior_candidates'],8)

    def test_actual_input_record_never_reads_diagnostic_target(self):
        k=torch.eye(3)[None];k[:,0,2]=258.5;k[:,1,2]=258.5
        trusted=dict(depth=torch.full((1,1,1),.1),valid=torch.ones(1,1,1,dtype=torch.bool),confidence=torch.ones(1,1,1))
        data=dict(k=k,w=torch.eye(4)[None],trusted=trusted,query_ids=torch.tensor([0]),matrix=torch.eye(4),
                  envelope=torch.ones(32,32,32,dtype=torch.bool),protected_pixels=torch.ones(1,1,1,dtype=torch.bool),domain={})
        with patch('trellis2_geometry_repair.experiments.guided_flow.data.read',side_effect=AssertionError('GT read')):
            result,meta=record(data,None,'input',5,42)
        self.assertFalse(meta['gt_used']);self.assertTrue(torch.equal(result['trusted']['valid'],trusted['valid']))

if __name__=='__main__':unittest.main()
