import argparse
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from ...tests.common import synthetic_sample
from ...contracts import ground_truth_prediction
from ...paired_dataset import sha256_file
from ..contracts import adapt_v1
from ..dataset import SceneDataset, write_manifest
from ..layer_predictor import GeometryConfig, RayLayerPredictor
from ..ray_geometry import CameraBundle
from ..losses import multiview_point_loss
from ..checkpoints import (save_checkpoint, read_checkpoint, load_state, initialize_ema,
                           restore_rng, update_ema)


class PersistenceTests(unittest.TestCase):
    def test_checkpoint_preserves_optimizer_ema_and_rng(self):
        torch.set_num_threads(2)
        config = GeometryConfig(input_channels=8, hidden_channels=8, evidence_channels=4, candidates=4)
        model = RayLayerPredictor(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        ema = initialize_ema(model)
        # Establish nonempty optimizer state on a genuine trainable tensor.
        p = next(model.parameters())
        p.square().sum().backward(); optimizer.step(); update_ema(model, ema)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, model=model, config=config, stage='lari', bindings={}, data_identity={},
                optimizer=optimizer, ema=ema, step=1, cursor={'epoch': 0, 'offset': 1}, args=argparse.Namespace())
            expected_random = torch.rand(4)
            checkpoint = read_checkpoint(path)
            clone = RayLayerPredictor(config)
            load_state(clone, checkpoint, config, stage='lari', bindings={})
            restored = torch.optim.AdamW(clone.parameters(), lr=.001)
            restored.load_state_dict(checkpoint['optimizer'])
            restore_rng(checkpoint['rng'])
            self.assertTrue(torch.equal(torch.rand(4), expected_random))
            self.assertTrue(torch.equal(next(clone.parameters()), next(model.parameters())))
            for original, new in zip(optimizer.state.values(), restored.state.values()):
                self.assertTrue(torch.equal(original['exp_avg'], new['exp_avg']))
            self.assertEqual(checkpoint['ema_state'].keys(), ema.keys())

    def test_dataset_hash_and_object_split(self):
        sample = adapt_v1(synthetic_sample(channels=1024))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.pt'; torch.save(sample, path)
            row = {'uid': sample['uid'], 'object_uid': sample['object_uid'], 'sample': path.name,
                   'sha256': sha256_file(path)}
            write_manifest(root / 'manifest.json', [row], {}, root)
            dataset = SceneDataset(root / 'manifest.json')
            self.assertEqual(dataset[0]['object_uid'], sample['object_uid'])
            with self.assertRaises(ValueError):
                dataset.assert_disjoint(dataset)
            sample['visible_target_mask'].zero_(); torch.save(sample, path)
            with self.assertRaises(ValueError):
                SceneDataset(root / 'manifest.json')[0]

    def test_gt_consistency_is_zero_with_different_camera_centers(self):
        sample = synthetic_sample()
        prediction = ground_truth_prediction(sample)
        loss, stats = multiview_point_loss(prediction, sample, CameraBundle.from_sample(sample))
        self.assertGreater(stats['cross_view_pairs'], 0)
        self.assertLess(float(loss), 1.e-9)

    def test_mixed_manifest_explicitly_adapts_v1_source(self):
        from ...contracts import LAYERED_SAMPLE_FORMAT
        sample = synthetic_sample(channels=1024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'v1.pt'; torch.save(sample, target)
            row = {'uid': sample['uid'], 'object_uid': sample['object_uid'], 'sample': str(target),
                   'sha256': sha256_file(target), 'source_sample_format': LAYERED_SAMPLE_FORMAT}
            write_manifest(root / 'mixed.json', [row], {}, root)
            adapted = SceneDataset(root / 'mixed.json')[0]
            self.assertEqual(adapted['data_domain'], 'legacy_clean')
            self.assertTrue(adapted['legacy_overflow_unknown'])


if __name__ == '__main__':
    unittest.main()
