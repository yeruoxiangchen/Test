import unittest

import numpy as np
import torch
import trimesh

from pose_aligned_layered_reconstruction.frozen_slat import canonical_coords, master_noise, noise_features, inference_inputs, INPUT_KEYS
from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records


class MeshEvaluationTests(unittest.TestCase):
    def test_coordinate_keyed_noise_independent_of_support_and_order(self):
        master = master_noise(42, 0, 'cpu')
        a = np.array([[0,1,2,3], [0,3,4,5]], np.int32)
        b = np.array([[0,3,4,5], [0,4,5,6], [0,1,2,3]], np.int32)
        self.assertTrue(torch.equal(noise_features(a,master), noise_features(b,master)[[2,0]]))
        self.assertTrue(torch.equal(master, master_noise(42,0,'cpu')))
        self.assertFalse(torch.equal(master, master_noise(43,0,'cpu')))

    def test_bad_coordinate_values_rejected(self):
        for value in (np.array([[1.,2.,3.]]), np.array([[64,1,2]]), np.array([[1,1,2,3]]),
                      np.array([[1,2,3], [1,2,3]])):
            with self.assertRaises(ValueError):
                canonical_coords(value)

    def test_inference_uses_only_observations(self):
        sample = {key: 'unused' for key in INPUT_KEYS}
        sample['masked_visual_patch_features'] = torch.randn(2,4,1024)
        sample.update(target_coords=torch.ones(2,3), layer_depths_gt=torch.ones(2), raw_visual_patch_features=torch.randn(2,4,1024))
        inputs, context = inference_inputs(sample,'cpu')
        self.assertEqual(set(inputs), set(INPUT_KEYS))
        self.assertEqual(len(context['cond']),2)
        self.assertTrue(torch.equal(context['cond'][0][0],sample['masked_visual_patch_features'][0]))
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in context['neg_cond']))

    def test_identical_mesh_exact_and_translated_mesh_not_realigned(self):
        box = trimesh.creation.box(extents=(.5,.5,.5))
        identity = surface_metrics(box,box,count=4000)
        self.assertEqual(identity['chamfer_l1'],0.)
        self.assertAlmostEqual(identity['normal_consistency'],1.)
        self.assertEqual(identity['fscore_0p01'],1.)
        translated=box.copy()
        translated.apply_translation([1.,0.,0.])
        different=surface_metrics(translated,box,count=4000)
        self.assertGreater(different['chamfer_l1'],.5)
        self.assertEqual(different['fscore_0p05'],0.)
        info=structure_metrics(box)
        self.assertEqual(info['component_count'],1)
        self.assertEqual(info['boundary_edge_count'],0)

    def records(self):
        return [dict(uid=uid,seed=seed,mode=mode,success=True,surface=dict(chamfer_l1=.03 if mode=='parent' else .02,
                fscore_0p02=.5 if mode=='parent' else .6))
                for uid in ('a','b') for seed in (42,43) for mode in ('parent','candidate')]

    def test_aggregation_direction_and_complete_pairs(self):
        records=self.records()
        result=aggregate_mesh_records(records,['a','b'])
        self.assertEqual(result['paired']['chamfer_l1']['win_objects'],2)
        self.assertAlmostEqual(result['paired']['chamfer_l1']['mean_improvement'],.01)
        self.assertAlmostEqual(result['paired']['fscore_0p02']['mean_improvement'],.1)
        for bad in (records[:-1],records+[records[0]],records+[dict(records[0],uid='other')]):
            with self.assertRaises(ValueError):
                aggregate_mesh_records(bad,['a','b'])
        records[0]['success']=False
        with self.assertRaises(ValueError):
            aggregate_mesh_records(records,['a','b'])
        records=self.records()
        records[0]['surface']['chamfer_l1']=float('nan')
        with self.assertRaises(ValueError):
            aggregate_mesh_records(records,['a','b'])


if __name__=='__main__':
    unittest.main()
