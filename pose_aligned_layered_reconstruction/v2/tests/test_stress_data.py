import unittest
import torch

from ...data_v2.build_legacy_occlusion import accept_visibility


class StressDataTests(unittest.TestCase):
    def test_difficulty_quota_rejects_easy_and_fully_hidden_scenes(self):
        policy = {'max_attempts': 64, 'min_strong_views': 4, 'strong_threshold': .5,
                  'min_mean_visible': .15, 'min_nonempty_views': 4}
        self.assertFalse(accept_visibility(torch.full((8,), .9), policy))
        self.assertFalse(accept_visibility(torch.zeros(8), policy))
        self.assertFalse(accept_visibility(torch.tensor([.2,.3,.4,.5,.6,.7,.8,.9]), policy))
        self.assertTrue(accept_visibility(torch.tensor([.2,.3,.4,.49,.6,.7,.8,.9]), policy))
        with self.assertRaises(ValueError):
            accept_visibility(torch.full((8,), float('nan')), policy)


if __name__ == '__main__': unittest.main()
