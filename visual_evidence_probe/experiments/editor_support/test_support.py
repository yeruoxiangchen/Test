"""CPU-only checks for the support intervention's preservation contract."""
import json
import unittest

import numpy as np

from .support import expand_support, plan_expansion


def center(cells):
    return (np.asarray(cells, dtype=float) + .5) / 64 - .5


def original(count=10):
    coords = np.column_stack((np.zeros(count, np.int32), np.arange(count, dtype=np.int32),
                              np.zeros((count, 2), np.int32)))
    features = np.arange(count * 8, dtype=np.float32).reshape(count, 8)
    return coords, features


class SupportTests(unittest.TestCase):
    def test_dual_budget_original_mapping_and_sorted_output(self):
        coords, features = original(10)
        coords, features = coords[::-1].copy(), features[::-1].copy()
        source = center([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        targets = center([[20, 0, 0], [21, 0, 0], [22, 0, 0]])
        c, f, added, r = plan_expansion(coords, features, source, targets, [1, 3, 2], max_added_tokens=128)
        self.assertEqual(r['token_budget'], 2)
        self.assertEqual(added.sum(), 2)
        np.testing.assert_array_equal(c[r['old_to_expanded']], coords)
        np.testing.assert_array_equal(f[r['old_to_expanded']], features)
        self.assertEqual(c.tolist(), sorted(c.tolist()))
        self.assertEqual(r['added_input_indices'], [1, 2])
        self.assertEqual(r['added_candidate_kind'], ['target', 'target'])
        self.assertFalse(r['mesh_preservation_checked'])
        json.dumps(r, allow_nan=False)

    def test_absolute_budget_and_fraction_floor(self):
        coords, features = original(6)
        source, targets = center([[0, 0, 0]]), center([[30, 0, 0]])
        _, _, mask, r = plan_expansion(coords, features, source, targets, [1], max_added_tokens=128)
        self.assertEqual(r['token_budget'], 1)
        self.assertEqual(mask.sum(), 1)
        _, _, mask, r = plan_expansion(coords, features, source, targets, [1], max_added_tokens=0)
        self.assertEqual(mask.sum(), 0)
        self.assertEqual(r['status'], 'budget_zero')

    def test_target_stage_precedes_paths_and_confidence_ties_are_stable(self):
        coords, features = original(20)
        source = center([[0, 0, 0], [1, 0, 0]])
        targets = center([[40, 0, 0], [50, 0, 0]])
        _, _, _, r = plan_expansion(coords, features, source, targets, [1, 1], max_fraction=.2)
        self.assertEqual(r['added_coordinates'][:2], [[40, 0, 0], [50, 0, 0]])
        self.assertEqual(r['added_input_indices'][:2], [0, 1])
        self.assertEqual(r['added_candidate_kind'][:2], ['target', 'target'])
        self.assertEqual(r['added_coordinates'][2:], [[20, 0, 0], [26, 0, 0]])

    def test_duplicate_cells_removed_and_original_features_used_only(self):
        coords, features = original(20)
        source = center([[0, 0, 0], [0, 0, 0]])
        targets = center([[40, 0, 0], [40, 0, 0]])
        c, f, added, r = plan_expansion(coords, features, source, targets, [2, 1])
        self.assertEqual(r['unique_new_candidates'], 2)
        self.assertEqual(added.sum(), 2)
        self.assertEqual(r['duplicate_new_candidate_samples'], 2)
        for row in np.flatnonzero(added):
            distances = np.linalg.norm(coords[:, 1:] - c[row, 1:], axis=1)
            nearest = np.argsort(distances, kind='stable')[:4]
            weights = 1 / distances[nearest]
            expected = (features[nearest] * (weights / weights.sum())[:, None]).sum(axis=0)
            np.testing.assert_allclose(f[row], expected, rtol=1.e-6)

    def test_knn_tie_break_independent_of_original_order(self):
        cells = np.array([[4, 5, 5], [6, 5, 5], [5, 4, 5], [5, 6, 5], [5, 5, 4]])
        coords = np.column_stack((np.zeros(5, np.int32), cells.astype(np.int32)))
        features = np.repeat(np.arange(5, dtype=np.float64)[:, None], 8, axis=1)
        points = center([[5, 5, 5]])
        a = plan_expansion(coords, features, points, points, [1])
        b = plan_expansion(coords[::-1], features[::-1], points, points, [1])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        # Lexicographic first four exclude x=6, and use equal unit weights.
        np.testing.assert_array_equal(a[1][a[2]][0], np.full(8, 2.25))

    def test_outside_and_nonfinite_targets_not_clipped_or_followed(self):
        coords, features = original(20)
        source = center([[0, 0, 0]] * 4)
        targets = np.array([[.5, 0, 0], [-.50001, 0, 0], [np.nan, 0, 0], [.49999, 0, 0]])
        c, _, added, r = plan_expansion(coords, features, source, targets, np.ones(4))
        self.assertEqual(r['unsupported_target_count'], 3)
        self.assertEqual(r['out_of_bounds_target_count'], 2)
        self.assertEqual(r['nonfinite_target_count'], 1)
        self.assertEqual(r['unsupported_target_indices'], [0, 1, 2])
        self.assertEqual(set(r['added_input_indices']), {3})
        self.assertTrue((c[added, 1:] < 64).all())

    def test_invalid_source_still_allows_valid_target_only(self):
        coords, features = original(10)
        _, _, added, r = plan_expansion(coords, features, [[np.nan, 0, 0]], center([[40, 0, 0]]), [1])
        self.assertEqual(added.sum(), 1)
        self.assertEqual(r['invalid_path_samples'], 2)
        self.assertEqual(r['added_candidate_kind'], ['target'])

    def test_upper_boundary_neighbor_quantizes_without_rounding_outside(self):
        coords, features = original(10)
        points = np.array([[np.nextafter(.5, -np.inf), 0, 0]])
        c, _, mask, r = plan_expansion(coords, features, points, points, [1])
        self.assertEqual(r['unsupported_target_count'], 0)
        self.assertEqual(c[mask].tolist(), [[0, 63, 32, 32]])

    def test_noop_is_independent_clone_and_preserves_unsorted_order(self):
        coords, features = original(10)
        coords, features = coords[::-1].copy(), features[::-1].copy()
        points = center([[1, 0, 0]])
        c, f, added, r = plan_expansion(coords, features, points, points, [1])
        self.assertEqual(r['status'], 'no_new_candidates')
        self.assertFalse(added.any())
        np.testing.assert_array_equal(c, coords)
        np.testing.assert_array_equal(f, features)
        self.assertFalse(np.shares_memory(c, coords))
        self.assertFalse(np.shares_memory(f, features))

    def test_shape_duplicate_and_batch_validation(self):
        coords, features = original()
        points = center([[30, 0, 0]])
        for invalid in (np.tile(coords[:1], (10, 1)), coords.astype(float), coords + [1, 0, 0, 0]):
            with self.subTest(invalid=invalid[:1].tolist()):
                with self.assertRaises(ValueError):
                    plan_expansion(invalid, features, points, points, [1])
        with self.assertRaises(ValueError):
            plan_expansion(coords, features[:, :7], points, points, [1])
        with self.assertRaises(ValueError):
            plan_expansion(coords, features, points, points, [1], max_fraction=-.1)

    def test_torch_adapter_cpu_fresh_tensor_and_exact_originals(self):
        import torch

        class FakeSparseTensor:
            def __init__(self, feats, coords, scale=(1, 1, 1)):
                self.feats, self.coords, self._scale = feats, coords, scale

        coords, features = original(10)
        z = FakeSparseTensor(torch.from_numpy(features).to(torch.bfloat16), torch.from_numpy(coords))
        points = torch.tensor(center([[30, 0, 0]]), dtype=torch.float32)
        expanded, added, r = expand_support(z, points, points, torch.ones(1))
        self.assertEqual(added.dtype, torch.bool)
        self.assertEqual(expanded.feats.dtype, z.feats.dtype)
        self.assertEqual(expanded.coords.dtype, z.coords.dtype)
        self.assertTrue(torch.equal(expanded.coords[r['old_to_expanded']], z.coords))
        self.assertTrue(torch.equal(expanded.feats[r['old_to_expanded']], z.feats))
        self.assertNotEqual(expanded.feats.data_ptr(), z.feats.data_ptr())
        self.assertNotEqual(expanded.coords.data_ptr(), z.coords.data_ptr())


if __name__ == '__main__':
    unittest.main()
