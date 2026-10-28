from __future__ import annotations

import unittest

import torch

from pose_aligned_layered_reconstruction.view_fusion import (
    EveryBlockConditionProjection,
    PoseViewAggregator,
    ZeroInitViewGate,
    aggregate_layer_evidence,
)


class ViewFusionTests(unittest.TestCase):
    def test_tiny_support_cannot_be_normalized_to_full_strength(self) -> None:
        encoded = torch.ones(2, 3, 4)
        appearance_weights = torch.full((2, 3), 0.5)
        strong, _ = aggregate_layer_evidence(
            encoded,
            appearance_weights=appearance_weights,
            target_support=torch.ones(2, 3),
        )
        weak, _ = aggregate_layer_evidence(
            encoded,
            appearance_weights=appearance_weights,
            target_support=torch.full((2, 3), 1.0e-4),
        )
        torch.testing.assert_close(weak, strong * 1.0e-4)

    def test_zero_view_gate_preserves_parent_aggregation_exactly(self) -> None:
        generator = torch.Generator().manual_seed(12)
        features = torch.randn((3, 7, 8), generator=generator)
        valid = torch.ones((3, 7), dtype=torch.bool)
        layer_features = torch.randn((3, 7, 12), generator=generator)
        aggregator = PoseViewAggregator(8)
        gate = ZeroInitViewGate(hidden_channels=4)
        baseline, _, _ = aggregator(features, valid)
        residual = gate(layer_features)
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        candidate, _, _ = aggregator(
            features, valid, logit_residual=residual
        )
        torch.testing.assert_close(candidate, baseline, rtol=0.0, atol=0.0)

    def test_layer_block_projection_is_exact_zero_at_initialization(self) -> None:
        adapter = EveryBlockConditionProjection(12, 16, 3)
        value = torch.randn((1, 9, 12))
        self.assertTrue(adapter.exact_zero())
        for index in range(3):
            self.assertEqual(int(torch.count_nonzero(adapter(index, value))), 0)


if __name__ == "__main__":
    unittest.main()
