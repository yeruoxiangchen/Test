import unittest
from unittest.mock import patch
import torch
from ... import layer_matching
from ...tests.common import synthetic_sample
from ...projection import canonical_query_points
from ...v2.conditioning import query_geometry
from ...v2.contracts import LayeredPredictionV2
from ..hssd_full_diagnostic_20261021.diagnose import oracle_prediction, replace_parts
from .interventions import features
from .common import VARIANTS


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.sample=synthetic_sample(channels=1024)
        self.gt=oracle_prediction(self.sample,'cpu')
        g=self.gt
        self.pred=LayeredPredictionV2(g.depths+.03,g.stop_logits*.08,g.target_logits*.05,g.confidence_logits*.01)
        self.points=canonical_query_points(16,self.sample['grid_transform'],device='cpu')[::41]

    def test_full_feature_native_parity(self):
        original=layer_matching._sample_prediction
        for v,p in [('predicted',self.pred),('oracle',self.gt)]:
            actual=features(self.sample,self.pred,self.gt,v,self.points)
            expected=query_geometry(self.sample,p,self.points,absolute_tolerance=.02,relative_tolerance=0.).features
            self.assertTrue(torch.equal(actual,expected),v)
        self.assertIs(layer_matching._sample_prediction,original)

    def test_interventions_preserve_predictions_and_restore_on_error(self):
        original=layer_matching._sample_prediction
        saved=[v.clone() for v in [self.pred.depths,self.pred.stop_logits,self.pred.target_logits,self.pred.confidence_logits]]
        for variant in VARIANTS:
            actual=features(self.sample,self.pred,self.gt,variant,self.points)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(actual.shape[-1],12)
        for a,b in zip(saved,[self.pred.depths,self.pred.stop_logits,self.pred.target_logits,self.pred.confidence_logits]):self.assertTrue(torch.equal(a,b))
        with patch('pose_aligned_layered_reconstruction.experiments.hssd_lari_causal_20261023.interventions.query_geometry',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):features(self.sample,self.pred,self.gt,'gt_depth',self.points)
        self.assertIs(layer_matching._sample_prediction,original)

    def test_component_isolation_and_invalid_depth_suffix(self):
        pp=tuple(torch.full((2,4,5),v) for v in [1.,.8,.7,.6])
        gg=tuple(torch.full((2,4,5),v) for v in [2.,1.,1.,1.])
        for v,changed in [('gt_depth',{0}),('gt_existence',{1}),('gt_ownership',{2}),('confidence_one',{3})]:
            mixed,tau=replace_parts(pp,gg,v)
            self.assertEqual(tau,.02)
            for i in set(range(4))-changed:self.assertIs(mixed[i],pp[i])
        gg[1][...,-1]=0.
        mixed,_=replace_parts(pp,gg,'gt_depth')
        self.assertTrue(torch.equal(mixed[0][...,-1],pp[0][...,-1]))


if __name__=='__main__':unittest.main()
