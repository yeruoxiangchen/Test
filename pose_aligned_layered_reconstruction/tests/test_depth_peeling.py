from __future__ import annotations

import unittest

import torch

from pose_aligned_layered_reconstruction.depth_peeling import (
    merge_target_and_occluder_layers,
)


class DepthPeelingTests(unittest.TestCase):
    def test_target_layers_remain_identified_behind_occluder(self) -> None:
        target = torch.tensor([3.0, 4.0, 0.0])[None, :, None, None]
        other = torch.tensor([1.0, 2.0, 5.0])[None, :, None, None]
        depths, stopping, ownership = merge_target_and_occluder_layers(
            target, other, output_layers=5
        )
        torch.testing.assert_close(
            depths.flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        torch.testing.assert_close(
            ownership.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
        )
        self.assertEqual(int(stopping.item()), 5)

    def test_invalid_suffix_is_zero(self) -> None:
        target = torch.tensor([2.0, 3.0])[None, :, None, None]
        other = torch.zeros((1, 1, 1, 1))
        depths, stopping, ownership = merge_target_and_occluder_layers(
            target, other, output_layers=4
        )
        self.assertEqual(tuple(depths.shape), (1, 4, 1, 1))
        self.assertEqual(int(stopping.item()), 2)
        self.assertEqual(float(depths[:, 2].item()), 0.0)
        self.assertEqual(float(ownership[:, 2].item()), 0.0)
        self.assertEqual(float(depths[:, 3].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
