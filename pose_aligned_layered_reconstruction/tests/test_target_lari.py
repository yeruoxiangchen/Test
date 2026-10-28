from __future__ import annotations

import unittest

import torch

from pose_aligned_layered_reconstruction.target_lari import (
    TargetConditionedLaRI,
    target_lari_loss,
)
from pose_aligned_layered_reconstruction.tests.common import synthetic_sample


class TargetLaRITests(unittest.TestCase):
    def test_prediction_is_ordered_and_loss_backpropagates(self) -> None:
        sample = synthetic_sample(channels=8)
        model = TargetConditionedLaRI(
            input_channels=8,
            hidden_channels=16,
            layers=5,
            decoder_blocks=1,
        )
        prediction = model(
            raw_patch_features=sample["raw_visual_patch_features"],
            masked_patch_features=sample["masked_visual_patch_features"],
            visible_target_mask=sample["visible_target_mask"],
            intrinsics=sample["intrinsics"],
            extrinsics=sample["extrinsics"],
            extrinsics_type=sample["extrinsics_type"],
            camera_forward_sign=sample["camera_forward_sign"],
            image_size=sample["image_size"],
        )
        self.assertTrue(bool((prediction.depths > 0).all().item()))
        self.assertEqual(prediction.depths.shape, (2, 5, 4, 4))
        self.assertTrue(bool((prediction.depths[:, 1:] > prediction.depths[:, :-1]).all()))
        loss, metrics = target_lari_loss(prediction, sample)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertIn("stopping_loss", metrics)

    def test_raw_branch_changes_prediction_without_changing_masked_identity(self) -> None:
        sample = synthetic_sample(channels=8)
        model = TargetConditionedLaRI(
            input_channels=8, hidden_channels=16, layers=5, decoder_blocks=1
        ).eval()
        kwargs = dict(
            masked_patch_features=sample["masked_visual_patch_features"],
            visible_target_mask=sample["visible_target_mask"],
            intrinsics=sample["intrinsics"],
            extrinsics=sample["extrinsics"],
            extrinsics_type=sample["extrinsics_type"],
            camera_forward_sign=sample["camera_forward_sign"],
            image_size=sample["image_size"],
        )
        first = model(
            raw_patch_features=sample["raw_visual_patch_features"], **kwargs
        )
        second = model(
            raw_patch_features=sample["raw_visual_patch_features"] + 1.0,
            **kwargs,
        )
        self.assertGreater(float((first.depths - second.depths).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
