from __future__ import annotations

import unittest

import torch

from pose_aligned_layered_reconstruction.contracts import LayeredPrediction
from pose_aligned_layered_reconstruction.layer_matching import (
    corrupt_layer_prediction,
    match_queries_to_layers,
)


def exact_prediction() -> LayeredPrediction:
    views, layers, side = 2, 5, 2
    depths = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])[None, :, None, None]
    depths = depths.expand(views, layers, side, side).clone()
    stop_logits = torch.full((views, layers + 1, side, side), -20.0)
    stop_logits[:, layers] = 20.0
    target_logits = torch.full_like(depths, -20.0)
    target_logits[:, 2:4] = 20.0
    confidence_logits = torch.full_like(depths, 20.0)
    return LayeredPrediction(
        depths, stop_logits, target_logits, confidence_logits
    ).validate()


class LayerMatchingTests(unittest.TestCase):
    def test_invalid_neighbor_depth_does_not_shift_valid_surface(self) -> None:
        depths = torch.tensor([[[[2.0, 100.0], [2.0, 100.0]]]])
        stopping = torch.full((1, 2, 2, 2), -20.0)
        stopping[:, 1, :, 0] = 20.0
        stopping[:, 0, :, 1] = 20.0
        logits = torch.full_like(depths, 20.0)
        prediction = LayeredPrediction(depths, stopping, logits, logits)
        match = match_queries_to_layers(
            prediction,
            image_grid=torch.zeros(1, 1, 2),
            camera_depth=torch.full((1, 1), 2.0),
            frustum_valid=torch.ones(1, 1, dtype=torch.bool),
            visible_target_mask=torch.ones(1, 2, 2),
        )
        self.assertAlmostEqual(float(match.matched_target_depth.item()), 2.0, places=5)
        self.assertGreater(float(match.target_support.item()), 0.49)

    def test_hidden_target_is_matched_behind_non_target_layers(self) -> None:
        prediction = exact_prediction()
        grid = torch.zeros((2, 1, 2))
        depth = torch.full((2, 1), 3.0)
        valid = torch.ones((2, 1), dtype=torch.bool)
        visible_mask = torch.zeros((2, 4, 4))
        match = match_queries_to_layers(
            prediction,
            image_grid=grid,
            camera_depth=depth,
            frustum_valid=valid,
            visible_target_mask=visible_mask,
            absolute_tolerance=0.05,
            relative_tolerance=0.0,
        )
        self.assertGreater(float(match.target_support.min()), 0.99)
        self.assertGreater(float(match.occluder_before.min()), 0.99)
        self.assertGreater(float(match.hidden_target_support.min()), 0.99)
        self.assertEqual(float(match.visible_target_support.max()), 0.0)

    def test_non_target_depth_does_not_become_target_support(self) -> None:
        prediction = exact_prediction()
        match = match_queries_to_layers(
            prediction,
            image_grid=torch.zeros((2, 1, 2)),
            camera_depth=torch.ones((2, 1)),
            frustum_valid=torch.ones((2, 1), dtype=torch.bool),
            visible_target_mask=torch.ones((2, 4, 4)),
            absolute_tolerance=0.05,
            relative_tolerance=0.0,
        )
        self.assertLess(float(match.target_support.max()), 1.0e-6)

    def test_layer_view_corruption_is_explicit(self) -> None:
        prediction = exact_prediction()
        prediction.depths[1] += 0.25
        corrupted = corrupt_layer_prediction(prediction, "layer_view_cyclic1")
        torch.testing.assert_close(corrupted.depths[0], prediction.depths[1])


if __name__ == "__main__":
    unittest.main()
