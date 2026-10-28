import unittest
import torch

from ...tests.common import synthetic_sample
from ...tests.test_native_ss_conditions import DummySSFlow
from ..ss import JointSSFlow
from ..layer_predictor import GeometryConfig


class JointConditionTests(unittest.TestCase):
    def test_zero_init_keeps_appearance_and_single_prediction_receives_gradients(self):
        torch.set_num_threads(2)
        sample = synthetic_sample(channels=1024)
        config = GeometryConfig(hidden_channels=16, evidence_channels=4, candidates=4, query_chunk=5)
        model = JointSSFlow(DummySSFlow(), config)
        predicted = model.predict_layers(sample, device=torch.device('cpu'), view_indices=None, source='predicted')
        original, _, _ = model.build_conditions(sample, device=torch.device('cpu'), use_layer_branch=False)
        # Inject the existing prediction; recomputation would fail this hook.
        def forbid_reprediction(*args, **kwargs):
            raise AssertionError('geometry was predicted twice')
        model.predict_layers = forbid_reprediction
        appearance, layer, _ = model.build_conditions(sample, device=torch.device('cpu'), layer_prediction=predicted)
        self.assertTrue(torch.equal(appearance, original))
        self.assertTrue(model.new_adapters_exact_zero())
        layer.square().mean().backward()
        self.assertIsNotNone(model.target_lari.gaps.weight.grad)
        self.assertTrue(torch.isfinite(model.target_lari.gaps.weight.grad).all())
        self.assertGreater(float(model.target_lari.gaps.weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
