from __future__ import annotations

import unittest

import numpy as np

from pose_aligned_layered_reconstruction.paired_preprocessing import (
    prepare_paired_arrays,
)


class PairedPreprocessingTests(unittest.TestCase):
    def test_raw_keeps_occluder_and_masked_removes_it_with_identical_crop(self) -> None:
        raw = np.zeros((80, 100, 3), dtype=np.uint8)
        raw[20:60, 30:70, 0] = 220  # visible target appearance
        raw[30:50, 55:80, 1] = 255  # non-target occluder/context
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:60, 30:55] = 255
        prepared = prepare_paired_arrays(
            [raw], [mask], resolution=64, crop_margin=2.0
        )
        raw_output = np.asarray(prepared.raw_images[0])
        masked_output = np.asarray(prepared.masked_images[0])
        self.assertEqual(raw_output.shape, masked_output.shape)
        self.assertGreater(int(raw_output[..., 1].max()), 0)
        self.assertEqual(int(masked_output[..., 1].max()), 0)
        self.assertGreater(int(masked_output[..., 0].max()), 0)
        self.assertEqual(prepared.visible_masks.shape, (1, 64, 64))

    def test_intrinsics_use_the_shared_crop_affine(self) -> None:
        raw = np.zeros((40, 60, 3), dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 20:40] = 255
        prepared = prepare_paired_arrays([raw], [mask], resolution=40)
        intrinsics = np.eye(3, dtype=np.float32)[None]
        transformed = prepared.transform_intrinsics(intrinsics)
        np.testing.assert_allclose(
            transformed, prepared.source_to_feature_affines, rtol=0, atol=0
        )

    def test_layer_stack_uses_the_same_crop_without_depth_blending(self) -> None:
        raw = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:7] = 255
        prepared = prepare_paired_arrays(
            [raw], [mask], resolution=8, crop_margin=1.0
        )
        layers = np.zeros((1, 2, 8, 10), dtype=np.float32)
        layers[0, 0, 2:6, 3:7] = 2.0
        layers[0, 1, 3:5, 4:6] = 4.0
        transformed = prepared.transform_view_layer_stack(
            layers, output_resolution=4
        )
        self.assertEqual(tuple(transformed.shape), (1, 2, 4, 4))
        self.assertEqual(set(np.unique(transformed[0, 0]).tolist()), {2.0})
        self.assertTrue(set(np.unique(transformed[0, 1]).tolist()) <= {0.0, 4.0})


if __name__ == "__main__":
    unittest.main()
