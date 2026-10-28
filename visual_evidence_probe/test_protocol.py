"""Check intervention pairing and inference/GT separation without GPU models."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .data import select_camera_order, input_for


class ProtocolTests(unittest.TestCase):
    def test_second_view_uses_only_camera_directions(self):
        extrinsics = torch.eye(4).repeat(8, 1, 1)
        for i, theta in enumerate(np.arange(8) * np.pi / 4):
            extrinsics[i, :3, 3] = torch.tensor([-np.cos(theta), -np.sin(theta), 0.])
        order = select_camera_order({'extrinsics_type': 'w2c', 'extrinsics': extrinsics})
        self.assertEqual(order[:2], [0, 4])
        self.assertEqual(sorted(order), list(range(8)))

    def test_source_marginals_camera_pairing_and_no_gt_leak(self):
        features = torch.arange(8).float()[:, None, None].expand(8, 4, 1024).clone()
        obs = dict(intrinsics=torch.eye(3).repeat(8, 1, 1), extrinsics=torch.eye(4).repeat(8, 1, 1),
                   image_size=(518, 518), grid_transform='identity', extrinsics_type='w2c',
                   camera_forward_sign=1., masked_visual_patch_features=features,
                   target_coords=torch.tensor([[1, 2, 3]]), layer_depths_gt=torch.ones(8, 5, 37, 37))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'a': features + 100, 'b': features + 200, 'gt': features + 300}, root / 'render_features.pt')
            with patch('visual_evidence_probe.data.object_data', return_value=({}, obs)), \
                 patch('visual_evidence_probe.data.folder', return_value=root):
                real = input_for(0, 'real2')
                repeat = input_for(0, 'repeat8')
                a, b = input_for(0, 'donor_a6'), input_for(0, 'donor_b6')
                ab, ba = input_for(0, 'mixed_ab6'), input_for(0, 'mixed_ba6')
            self.assertTrue(torch.equal(repeat['masked_visual_patch_features'], features[[0, 1] * 4]))
            self.assertEqual(len(real['intrinsics']), 2)
            self.assertTrue(torch.equal(a['intrinsics'], ab['intrinsics']))
            self.assertTrue(torch.equal(a['extrinsics'], ba['extrinsics']))
            self.assertTrue(torch.equal(a['masked_visual_patch_features'] + b['masked_visual_patch_features'],
                                        ab['masked_visual_patch_features'] + ba['masked_visual_patch_features']))
            for sample in (real, repeat, a, b, ab, ba):
                self.assertNotIn('target_coords', sample)
                self.assertNotIn('layer_depths_gt', sample)
                self.assertNotIn('target_mesh_binding', sample)
                self.assertEqual(len(sample['slat_condition']['cond']), len(sample['intrinsics']))


if __name__ == '__main__':
    unittest.main()
