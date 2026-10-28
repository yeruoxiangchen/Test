from __future__ import annotations

import unittest

import torch

from pose_aligned_layered_reconstruction.contracts import (
    ground_truth_prediction,
    validate_paired_sample,
)
from pose_aligned_layered_reconstruction.tests.common import synthetic_sample


class PairedContractTests(unittest.TestCase):
    def test_valid_paired_sample(self) -> None:
        metadata = validate_paired_sample(
            synthetic_sample(), require_labels=True, require_ss_target=True
        )
        self.assertEqual(metadata["views"], 2)
        self.assertEqual(metadata["layers"], 5)

    def test_view_mismatch_is_rejected(self) -> None:
        sample = synthetic_sample()
        sample["raw_visual_patch_features"] = sample[
            "raw_visual_patch_features"
        ][:1]
        with self.assertRaisesRegex(ValueError, "raw/masked"):
            validate_paired_sample(sample)

    def test_target_cannot_occupy_invalid_layer(self) -> None:
        sample = synthetic_sample()
        sample["ray_stop_index_gt"].fill_(2)
        sample["layer_target_gt"].zero_()
        sample["layer_target_gt"][:, 3] = 1.0
        with self.assertRaisesRegex(ValueError, "cannot own"):
            validate_paired_sample(sample, require_labels=True)

    def test_oracle_sanitizes_invalid_suffix_without_changing_valid_depths(self) -> None:
        sample = synthetic_sample()
        sample["ray_stop_index_gt"].fill_(2)
        sample["layer_target_gt"][:, 2:] = 0.0
        sample["layer_depths_gt"][:, 2:] = 0.0
        prediction = ground_truth_prediction(sample)
        torch.testing.assert_close(
            prediction.depths[:, :2], sample["layer_depths_gt"][:, :2]
        )
        self.assertTrue(bool((prediction.depths[:, 1:] > prediction.depths[:, :-1]).all()))
        valid_probability = prediction.valid_probabilities()
        self.assertGreater(float(valid_probability[:, :2].min()), 0.999)
        self.assertLess(float(valid_probability[:, 2:].max()), 0.001)


if __name__ == "__main__":
    unittest.main()
