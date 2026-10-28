"""Focused checks for held-out provenance and a non-contaminating queue."""
import unittest
from unittest.mock import patch
from . import common as c
from .evaluate import validate_checkpoint


class ProtocolTests(unittest.TestCase):
    def test_actual_split_disjoint(self):
        p=c.plan();self.assertEqual(len(p['objects']),12)
        for field in ('scene','uid','asset_family','candidate_geometry_hash'):
            train={r[field] for r in p['objects'] if r['split']=='train'}
            val={r[field] for r in p['objects'] if r['split']=='val'}
            self.assertFalse(train&val)

    def test_heldout_loader_never_loads_targets(self):
        from pose_constrained_trellis2.data import PoseTrellis2Dataset
        with patch('pose_constrained_trellis2.data.load_ss_target',side_effect=AssertionError('GT read')),patch('pose_constrained_trellis2.data.load_shape_target',side_effect=AssertionError('GT read')):
            sample=PoseTrellis2Dataset(c.DATA/'val.json',stage='inference')[0]
        self.assertNotIn('target',sample);self.assertNotIn('target_feats',sample)
        self.assertEqual(sample['row']['split'],'val')

    def test_checkpoint_stays_bound_to_training(self):
        from pose_constrained_trellis2.data import PoseTrellis2Dataset
        train=PoseTrellis2Dataset(c.DATA/'train.json',stage='ss').identity()
        val=PoseTrellis2Dataset(c.DATA/'val.json',stage='ss').identity()
        good={'data_identity':train,'step':400,'context_mode':'first'}
        validate_checkpoint(good,train,400)
        with self.assertRaises(RuntimeError):validate_checkpoint(good,val,400)
        with self.assertRaises(ValueError):validate_checkpoint(good,train,2)

    def test_queue_covers_frozen_denominator(self):
        from .runner import stages
        jobs=stages('all')
        evals=[j for j in jobs if j[0].startswith('eval_')]
        self.assertEqual(len(evals),48)
        self.assertEqual(len(set(x[0] for x in jobs)),len(jobs))
        for name,args,_,_ in jobs:
            if name.startswith('train_'):
                self.assertIn(str(c.DATA/'train.json'),args)
                self.assertNotIn(str(c.DATA/'all.json'),args)
                self.assertNotIn('--indices',args)

    def test_storage_and_input_only_pairs(self):
        self.assertTrue(c.OUT.is_relative_to(c.ROOT));self.assertTrue(c.DATA.is_relative_to(c.ROOT))
        doc=c.read(c.DATA/'pair_proposals_v2.json')
        for r in doc['groups']:
            self.assertGreaterEqual(len(r['selected_frames']),2)
            self.assertNotIn('gt_qualification',r)
            self.assertEqual(doc['endpoint_assets'][r['a']]['split'],doc['endpoint_assets'][r['b']]['split'])
        train={doc['endpoint_assets'][u]['asset_family'] for u in doc['endpoint_assets'] if doc['endpoint_assets'][u]['split']=='train'}
        val={doc['endpoint_assets'][u]['asset_family'] for u in doc['endpoint_assets'] if doc['endpoint_assets'][u]['split']=='val'}
        self.assertFalse(train&val)

    def test_gpu_does_not_take_busy_device(self):
        from .runner import choose_gpu
        idle='0, uuid0, 266, 23862, 0\n1, uuid1, 19000, 5000, 0\n'
        with patch('subprocess.check_output',side_effect=[idle,'uuid0, 256\nuuid1, 19000\n']):
            self.assertEqual(choose_gpu(),0)
        with patch('subprocess.check_output',side_effect=[idle,'uuid0, 18000\nuuid1, 19000\n']):
            self.assertIsNone(choose_gpu())


if __name__=='__main__':unittest.main(verbosity=2)
