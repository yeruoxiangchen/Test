"""Behavior checks for relation gradients, geometry, synchrony, and target isolation."""
import tempfile
import unittest
from unittest.mock import patch
import torch
from ...conditioning import EveryBlockCondition
from ...scene_relations.model import RelationalBlockCondition, summarize_state, bind_neighbors
from ...scene_relations.sampling import synchronous_step
from ...scene_relations.data import GroupDataset


class RelationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.block = RelationalBlockCondition(EveryBlockCondition(8, 12, 2), stage='ss',
                                              condition_channels=8, model_channels=12)
        self.condition = torch.randn(1, 8, 8)
        self.context = (torch.randn(8, 8), torch.randn(8, 3), torch.eye(4),
                        [{'uid': 'neighbor', 'features': torch.randn(4, 8),
                          'world_points': torch.randn(4, 3), 'appearance': torch.randn(8)}])

    def test_zero_init_preserves_base_and_receives_gradient(self):
        self.block.context = self.context
        result = self.block(1, self.condition)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertTrue(self.block.exact_zero())
        (result - 1).square().mean().backward()
        output = self.block.relation.outputs['1']
        self.assertGreater(float(output.weight.grad.abs().sum()), 0)
        with torch.no_grad(): output.weight.add_(-.1 * output.weight.grad)
        self.assertFalse(self.block.exact_zero())

    def test_neighbor_state_and_position_affect_condition(self):
        a = self.block.relation.attend(self.condition, self.context)
        own, pos, T, neighbors = self.context
        changed = [{**neighbors[0], 'features': neighbors[0]['features'] + 2}]
        b = self.block.relation.attend(self.condition, (own, pos, T, changed))
        moved = [{**neighbors[0], 'world_points': neighbors[0]['world_points'] + 1}]
        d = self.block.relation.attend(self.condition, (own, pos, T, moved))
        self.assertFalse(torch.allclose(a, b)); self.assertFalse(torch.allclose(a, d))

    def test_common_world_transform_cancels(self):
        own, pos, T, neighbors = self.context
        a = self.block.relation.attend(self.condition, self.context)
        G = torch.tensor([[0., -1, 0, 2], [1., 0, 0, -3], [0., 0, 1, 1], [0., 0, 0, 1]])
        changed = [{**n, 'world_points': n['world_points'] @ G[:3, :3].T + G[:3, 3]} for n in neighbors]
        b = self.block.relation.attend(self.condition, (own, pos, G @ T, changed))
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_neighbor_permutation_is_invariant(self):
        own, pos, T, neighbors = self.context
        other = {**neighbors[0], 'uid': 'second', 'features': neighbors[0]['features'] - 1}
        a = self.block.relation.attend(self.condition, (own, pos, T, neighbors + [other]))
        b = self.block.relation.attend(self.condition, (own, pos, T, [other] + neighbors))
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_synchronous_updates_do_not_read_updated_neighbors(self):
        result = synchronous_step([1, 2], lambda i, old: old[i] + old[1 - i])
        self.assertEqual(result, [3, 3])

    def test_inference_never_opens_training_labels(self):
        from . import common as c
        # Pure in-memory manifest; no filesystem cache/temp output.
        document = {'groups': [{'id': 'g', 'scene': 's', 'split': 'val',
                               'members': [{'uid': 'a', 'scene': 's', 'split': 'val',
                                            'pose_file': '/unused', 'ss_target_file': '/forbidden'}]}]}
        with patch('pathlib.Path.read_text', return_value=__import__('json').dumps(document)), \
             patch('pose_constrained_trellis2.scene_relations.data.load_pose_payload', return_value={'dino_features': torch.zeros(1)}), \
             patch('pose_constrained_trellis2.scene_relations.data.load_ss_target', side_effect=AssertionError('GT read')), \
             patch('pose_constrained_trellis2.scene_relations.data.load_shape_target', side_effect=AssertionError('GT read')):
            sample = GroupDataset('/unused.json', 'inference')[0][0]
        self.assertNotIn('target', sample); self.assertNotIn('target_feats', sample)

    def test_summary_uses_current_state_and_detaches_input(self):
        state = torch.randn(1, 8, 4, 4, 4, requires_grad=True)
        sample = {'uid': 'a', 'dino_features': torch.randn(2, 3, 8), 'object_to_world': torch.eye(4),
                  'target': torch.full_like(state, 999.)}
        result = summarize_state(state, sample, 'ss')
        self.assertFalse(result['features'].requires_grad)
        torch.testing.assert_close(result['features'], state.detach().flatten(2).transpose(1, 2)[0])
        self.assertEqual(result['uid'], 'a')


class CompleteSupportTests(unittest.TestCase):
    def arrays(self, count):
        import numpy as np
        coords = np.array(np.unravel_index(np.arange(count), (32, 32, 32))).T.astype(np.int32)
        feats = np.linspace(-1, 1, count * 32, dtype=np.float32).reshape(count, 32)
        return coords, feats

    def pack(self, coords, feats):
        import io, numpy as np
        out = io.BytesIO(); np.savez(out, coords=coords, feats=feats); out.seek(0); return out

    def test_dense_target_preserves_all_10457_tokens(self):
        from ...scene_relations.shape_support import load_shape_target
        coords, feats = self.arrays(10457)
        loaded, _ = load_shape_target(self.pack(coords, feats), 'dense')
        self.assertTrue(torch.equal(loaded, torch.from_numpy(coords)))

    def test_existing_target_normalization_is_unchanged(self):
        from ...scene_relations.shape_support import load_shape_target
        from ...data import load_shape_target as old_loader
        coords, feats = self.arrays(117)
        current = load_shape_target(self.pack(coords, feats), 'small')
        old = old_loader(self.pack(coords, feats), 'small')
        for a, b in zip(current, old): self.assertTrue(torch.equal(a, b))

    def test_invalid_support_is_still_rejected(self):
        import numpy as np
        from ...scene_relations.shape_support import validate_shape
        coords, feats = self.arrays(10457)
        with self.assertRaises(ValueError): validate_shape(coords, feats, max_tokens=8192)
        invalid = coords.copy(); invalid[-1] = invalid[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'): validate_shape(invalid, feats)
        invalid = coords.copy(); invalid[-1, 0] = 32
        with self.assertRaisesRegex(ValueError, 'outside'): validate_shape(invalid, feats)
        invalid_feats = feats.copy(); invalid_feats[0, 0] = np.nan
        with self.assertRaises(ValueError): validate_shape(coords, invalid_feats)

    def test_inference_retains_full_native_grid(self):
        from ...scene_relations.shape_support import ss_latent_to_shape_coords
        coords = ss_latent_to_shape_coords(torch.zeros(1, 8, 16, 16, 16),
                    lambda _: torch.ones(1, 1, 64, 64, 64))
        self.assertEqual(tuple(coords.shape), (32768, 4))
        self.assertEqual(len(torch.unique(coords, dim=0)), 32768)

    def test_migration_accepts_only_reviewed_receipt(self):
        from . import common as c
        row = {'runtime_identity': 'old', 'artifacts': []}
        migration = {'old_identity': 'old', 'new_identity': 'new', 'receipt_digests': {'checked': c.digest(row)}}
        with patch.object(c, 'read', return_value=migration):
            c.check_receipt_identity('checked', row, 'new')
            with self.assertRaises(AssertionError): c.check_receipt_identity('other', row, 'new')
            with self.assertRaises(AssertionError): c.check_receipt_identity('checked', {**row, 'extra': 1}, 'new')


if __name__ == '__main__': unittest.main(verbosity=2)
