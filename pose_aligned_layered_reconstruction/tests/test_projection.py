from __future__ import annotations

import unittest
import torch

from pose_aligned_layered_reconstruction.projection import resize_sampling_grid, sample_maps


class PatchCenterTests(unittest.TestCase):
    def test_layer_values_are_sampled_at_dino_patch_centers(self) -> None:
        # A 28px image has two 14px patches, centered at 6.5 and 20.5.
        centers = torch.tensor([6.5, 20.5])
        pixels = torch.stack((centers, torch.full_like(centers, 6.5)), -1)[None]
        image_grid = 2 * pixels / 27 - 1
        layer_grid = resize_sampling_grid(
            image_grid, image_size=(28, 28), map_size=(2, 2)
        )
        layer_map = torch.tensor([[[[2.0, 7.0], [2.0, 7.0]]]])
        sampled = sample_maps(layer_map, layer_grid)
        torch.testing.assert_close(sampled, torch.tensor([[[2.0, 7.0]]]))

    def test_native_resolution_grid_is_unchanged(self) -> None:
        grid = torch.rand(2, 4, 2) * 2 - 1
        result = resize_sampling_grid(grid, image_size=(518, 518), map_size=(518, 518))
        torch.testing.assert_close(grid, result, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
