import unittest
import numpy as np
import torch

from pose_aligned_layered_reconstruction.evaluate_ss_occupancy import occupancy_metrics, aggregate_records
from pose_aligned_layered_reconstruction.projection import build_ss_projection


class RoundOneTests(unittest.TestCase):
    def test_sparse_overlap_and_duplicates(self):
        gt = np.array([[1, 1, 1], [2, 2, 2]])
        pred = np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 3, 3, 3]])
        result = occupancy_metrics(pred, gt)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)

    def test_empty_prediction_is_failure_not_missing_record(self):
        result = occupancy_metrics(np.empty((0, 4)), np.array([[1, 1, 1]]))
        self.assertEqual(result["iou"], 0)
        self.assertEqual(result["recall"], 0)

    def test_object_paired_average(self):
        records = []
        for uid in ("a", "b"):
            for seed in (42, 43):
                for mode, value in (("parent", .3), ("candidate", .4)):
                    records.append(dict(uid=uid, seed=seed, mode=mode, **dict.fromkeys(
                        ("iou", "precision", "recall", "f1", "count_ratio"), value)))
        result = aggregate_records(records)["paired"]
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["object_iou_win_rate"], 1)
        self.assertAlmostEqual(result["iou_gain"], .1)

    def test_layer_camera_axis_does_not_mutate_parent_projection(self):
        k = torch.tensor([[[400., 0., 259.], [0., 400., 259.], [0., 0., 1.]]])
        t = torch.eye(4)[None]
        t[:, 2, 3] = 2
        sample = dict(intrinsics=k, extrinsics=t, extrinsics_type="w2c", camera_forward_sign=1.,
                      image_size=(518, 518), grid_transform="identity",
                      masked_visual_patch_features=torch.zeros(1, 1369, 1024))
        parent = build_ss_projection(sample, device=torch.device("cpu"))
        sample["layer_camera_y_sign"] = -1.
        repeated = build_ss_projection(sample, device=torch.device("cpu"))
        layered = build_ss_projection(sample, device=torch.device("cpu"), for_layer_branch=True)
        self.assertTrue(torch.equal(parent["image_grid"], repeated["image_grid"]))
        self.assertFalse(torch.equal(parent["image_grid"], layered["image_grid"]))
        self.assertTrue(torch.equal(parent["camera_depth"], layered["camera_depth"]))
        self.assertEqual(float(k[0, 1, 1]), 400.)


if __name__ == "__main__":
    unittest.main()
