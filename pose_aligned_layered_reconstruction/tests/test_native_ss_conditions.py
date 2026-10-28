from __future__ import annotations

import unittest

import torch
from torch import nn

from pose_aligned_layered_reconstruction.native_ss import LayeredNativeSSFlow
from pose_aligned_layered_reconstruction.tests.common import synthetic_sample


class DummySSFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resolution = 16
        self.in_channels = 8
        self.out_channels = 8
        self.patch_size = 1
        self.model_channels = 16
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])


class NativeSSConditionTests(unittest.TestCase):
    def test_predicted_conditions_do_not_require_hidden_geometry_labels(self) -> None:
        sample = synthetic_sample(channels=1024)
        for key in ("layer_depths_gt", "ray_stop_index_gt", "layer_target_gt", "target"):
            sample.pop(key)
        model = LayeredNativeSSFlow(
            DummySSFlow(), layer_channels=8, lari_hidden_channels=16
        ).eval()
        with torch.no_grad():
            _, layer, _ = model.build_conditions(
                sample, device=torch.device("cpu"), layer_source="predicted"
            )
        self.assertEqual(tuple(layer.shape), (1, 4096, 8))
        self.assertTrue(bool(torch.isfinite(layer).all()))

    def test_layer_branch_preserves_parent_appearance_at_zero_init(self) -> None:
        sample = synthetic_sample(channels=1024)
        model = LayeredNativeSSFlow(
            DummySSFlow(),
            condition_channels=1024,
            layer_channels=8,
            lari_hidden_channels=16,
            lari_layers=5,
        ).eval()
        parent, _, _ = model.build_conditions(
            sample,
            device=torch.device("cpu"),
            use_layer_branch=False,
        )
        candidate, layer, stats = model.build_conditions(
            sample,
            device=torch.device("cpu"),
            use_layer_branch=True,
            layer_source="oracle",
        )
        torch.testing.assert_close(candidate, parent, rtol=0.0, atol=0.0)
        self.assertIsNotNone(layer)
        self.assertTrue(model.new_adapters_exact_zero())
        self.assertIn("hidden_target_support_mean", stats)


if __name__ == "__main__":
    unittest.main()
