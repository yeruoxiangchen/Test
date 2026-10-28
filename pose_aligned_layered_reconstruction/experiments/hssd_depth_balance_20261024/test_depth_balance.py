import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import torch
from ...v2.contracts import LayeredPredictionV2
from ...v2.losses import layer_loss
from .losses import depth_groups, supervised_loss
from . import runner
from .common import read


def fixture():
    # Three rays: target only, foreground non-target then target, non-target only.
    depth=torch.tensor([[[[.3,.2,.5]],[[.7,.6,.8]],[[.9,1.,1.1]]]],requires_grad=True)
    pred=LayeredPredictionV2(depth,torch.zeros(1,4,1,3,requires_grad=True),
        torch.zeros(1,3,1,3,requires_grad=True),torch.zeros(1,3,1,3,requires_grad=True))
    sample={'layer_depths_gt':torch.ones_like(depth)*.4,
        'layer_target_gt':torch.tensor([[[[1.,0.,0.]],[[1.,1.,0.]],[[0.,0.,0.]]]]),
        'ray_stop_index_gt':torch.tensor([[[2,2,1]]])}
    return pred,sample


class DepthTests(unittest.TestCase):
    def test_disjoint_partition_and_hidden_definition(self):
        valid=torch.tensor([[[[1,1,1]],[[1,1,0]],[[0,0,0]]]],dtype=torch.bool)
        _,s=fixture();g=depth_groups(valid,s['layer_target_gt'])
        self.assertEqual([int(x.sum()) for x in g.values()],[2,2,1])
        self.assertTrue(torch.equal(sum(x.long() for x in g.values()),valid.long()))

    def test_uniform_exact_value_and_gradient(self):
        p,s=fixture();a,_=layer_loss(p,s);b,_=supervised_loss(p,s,'uniform')
        self.assertTrue(torch.equal(a,b))
        values=[p.depths,p.stop_logits,p.target_logits,p.confidence_logits]
        ga=torch.autograd.grad(a,values,retain_graph=True);gb=torch.autograd.grad(b,values)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))

    def test_only_depth_direct_gradient_changes(self):
        p,s=fixture();a,_=supervised_loss(p,s,'uniform');b,stats=supervised_loss(p,s,'balanced')
        values=[p.depths,p.stop_logits,p.target_logits,p.confidence_logits]
        ga=torch.autograd.grad(a,values,retain_graph=True);gb=torch.autograd.grad(b,values)
        self.assertFalse(torch.equal(ga[0],gb[0]))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga[1:],gb[1:])))
        self.assertEqual(stats['active_depth_groups'],3)
        self.assertAlmostEqual(float(stats['balanced_depth']),(.0125+.025+.02)/3,places=6)

    def test_masked_and_empty_groups_are_finite(self):
        p,s=fixture();s['supervision_valid_mask']=torch.zeros(1,1,1,3)
        loss,stats=supervised_loss(p,s,'balanced')
        self.assertEqual(stats['active_depth_groups'],0)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(float(p.depths.grad.abs().sum()),0.)

    def test_repeating_background_does_not_dilute_balanced_depth(self):
        p,s=fixture()
        # Repeat the non-target-only ray many times, preserving its depth error.
        ids=torch.tensor([0,1]+[2]*20)
        q=LayeredPredictionV2(*(getattr(p,k).index_select(-1,ids) for k in ['depths','stop_logits','target_logits','confidence_logits']))
        t={k:v.index_select(-1,ids) for k,v in s.items()}
        # Make the two non-target absolute errors equal, then duplicate one ray.
        p.depths.data[0,0,0,1]=.3;q.depths.data[0,0,0,1]=.3
        _,m=supervised_loss(p,s,'balanced');_,n=supervised_loss(q,t,'balanced')
        self.assertAlmostEqual(float(m['balanced_depth']),float(n['balanced_depth']),places=6)
        self.assertNotAlmostEqual(float(m['uniform_depth']),float(n['uniform_depth']),places=5)

    def test_single_group_renormalizes_to_uniform(self):
        p,s=fixture();s['layer_target_gt']=torch.zeros_like(s['layer_target_gt'])
        a,_=supervised_loss(p,s,'uniform');b,m=supervised_loss(p,s,'balanced')
        self.assertEqual(m['active_depth_groups'],1)
        self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=0))

    def test_run_stops_before_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            with patch.object(runner,'OUT',Path(folder)),patch.object(runner,'require_preflight'),patch.object(runner,'validate_training'),patch.object(runner,'child',side_effect=lambda stage,cmd:calls.append((stage,cmd))):
                runner.queue('run')
            self.assertEqual([x[0] for x in calls],['lari_uniform','lari_balanced','lari_audit','ss_train_uniform','ss_train_balanced',
                'ss_base','ss_appearance_control','ss_oracle_point','ss_uniform','ss_balanced','ss_summary'])
            self.assertFalse(any('mesh' in x[0] for x in calls))
            self.assertEqual(read(Path(folder)/'status.json')['status'],'ss_complete_review_required')

    def test_failed_stage_never_launches_following_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def fail(stage,cmd):
                calls.append(stage);raise RuntimeError('intentional test failure')
            with patch.object(runner,'OUT',Path(folder)),patch.object(runner,'require_preflight'),patch.object(runner,'child',side_effect=fail):
                with self.assertRaises(RuntimeError):runner.queue('run')
            self.assertEqual(calls,['lari_uniform'])
            self.assertEqual(read(Path(folder)/'status.json')['status'],'failed')


if __name__=='__main__':unittest.main()
