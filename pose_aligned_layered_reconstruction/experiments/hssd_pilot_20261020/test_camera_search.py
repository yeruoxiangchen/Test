"""CPU regressions for narrow-room camera search and preservation of prior inputs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from . import scenes
from .common import OUT, SOURCE, object_dir, plan


def pool(count):
    return (np.repeat(np.eye(3)[None], count, 0),
            np.repeat(np.eye(4)[None], count, 0),
            [[float(i), 1.25, 0.] for i in range(count)])


def records(count, valid=None):
    valid = count if valid is None else valid
    return [{'index': i, 'eligible': i < valid, 'visible_fraction': i / max(count, 1)}
            for i in range(count)]


class CameraSearch(unittest.TestCase):
    def test_geometric_and_visibility_failures_both_trigger_fallback(self):
        assess = Mock(side_effect=[records(10, valid=7), records(12)])
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch.object(scenes, 'candidates', side_effect=[pool(4), pool(10), pool(12)]) as generate:
                k, w, _, chosen, name = scenes.select_input_cameras(
                    {'uid': 'fixture'}, None, assess, Path(folder))
            audit = json.loads((Path(folder) / 'camera_search.json').read_text())
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(assess.call_count, 2)  # No render when geometry already fails.
        self.assertEqual(name, 'dense_75deg')
        self.assertEqual(len(set(chosen)), 8)
        self.assertEqual((len(k), len(w)), (8, 8))
        self.assertEqual([r['eligible_candidates'] for r in audit['attempts']], [0, 7, 12])
        self.assertTrue(audit['passed'])

    def test_no_valid_pool_still_stops_and_keeps_failed_attempts(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch.object(scenes, 'candidates', return_value=pool(12)):
                with self.assertRaisesRegex(ValueError, 'Insufficient valid input cameras'):
                    scenes.select_input_cameras(
                        {'uid': 'fixture'}, None, lambda *args: records(12, valid=7), Path(folder))
            audit = json.loads((Path(folder) / 'camera_search.json').read_text())
        self.assertFalse(audit['passed'])
        self.assertEqual(len(audit['attempts']), len(scenes.CAMERA_SEARCHES))
        self.assertNotIn('selected_candidates', audit)

    def test_original_success_does_not_use_fallback(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch.object(scenes, 'candidates', return_value=pool(12)) as generate:
                *_, name = scenes.select_input_cameras(
                    {'uid': 'fixture'}, None, lambda *args: records(12), Path(folder))
        generate.assert_called_once()
        self.assertEqual(name, 'original_60deg')

    @unittest.skipUnless((OUT / 'plan.json').exists(), 'HSSD pilot assets required')
    def test_all_frozen_targets_have_valid_geometric_candidates(self):
        regions = json.loads((SOURCE / 'HSSD-hab/semantics/scenes/102344049.semantic_config.json')
                             .read_text())['region_annotations']
        failed_original = []
        for row in plan()['plan']['objects']:
            with self.subTest(uid=row['uid']):
                mesh = trimesh.load(row['mesh']['path'], force='mesh', process=False)
                for index, search in enumerate(scenes.CAMERA_SEARCHES):
                    options = {key: val for key, val in search.items() if key != 'name'}
                    k, w, eyes = scenes.candidates(row, mesh, minimum=0, **options)
                    if index == 0 and len(k) < 8:
                        failed_original.append(row['instance'])
                    if len(k) >= 8:
                        break
                self.assertGreaterEqual(len(k), 8)
                self.assertEqual(len(np.unique(np.asarray(eyes), axis=0)), len(k))
                canonical = np.asarray(row['canonical_from_world'])
                center = np.linalg.inv(canonical)[:3, 3]
                rooms = [np.array(r['poly_loop'])[:, [0, 2]] for r in regions
                         if scenes.in_polygon(center[[0, 2]], np.array(r['poly_loop'])[:, [0, 2]])]
                corners = trimesh.bounds.corners(mesh.bounds)
                for intrinsic, extrinsic, eye in zip(k, w, eyes):
                    self.assertTrue(any(scenes.in_polygon(np.array(eye)[[0, 2]], room) for room in rooms))
                    xyz = corners @ extrinsic[:3, :3].T + extrinsic[:3, 3]
                    uvw = xyz @ intrinsic.T
                    uv = uvw[:, :2] / uvw[:, 2:]
                    self.assertGreater(float(xyz[:, 2].min()), .05 - 1.e-5)
                    self.assertGreaterEqual(float(uv.min()), 8 - 1.e-3)
                    self.assertLessEqual(float(uv.max()), 510 + 1.e-3)
                    self.assertAlmostEqual(float(np.linalg.det(extrinsic[:3, :3])), 1., places=5)
        self.assertEqual(failed_original, [51, 56, 108])

    @unittest.skipUnless((OUT / 'plan.json').exists(), 'HSSD pilot assets required')
    def test_three_completed_camera_sets_are_bitwise_unchanged(self):
        for row in plan()['plan']['objects'][:3]:
            with self.subTest(uid=row['uid']), tempfile.TemporaryDirectory() as folder:
                root = object_dir(row)
                previous = json.loads((root / 'camera_candidates.json').read_text())
                previous_receipt = json.loads((root / 'scene_receipt.json').read_text())
                mesh = trimesh.load(row['mesh']['path'], force='mesh', process=False)
                assess = Mock(return_value=previous)
                with redirect_stdout(io.StringIO()):
                    k, w, _, chosen, name = scenes.select_input_cameras(
                        row, mesh, assess, Path(folder))
                assess.assert_called_once()
                self.assertEqual(len(assess.call_args.args[0]), len(previous))
                self.assertEqual(name, 'original_60deg')
                self.assertEqual(chosen, previous_receipt['selected_candidates'])
                with np.load(root / 'camera.npz') as old:
                    np.testing.assert_array_equal(k, old['intrinsics'])
                    np.testing.assert_array_equal(w, old['extrinsics'])


if __name__ == '__main__':
    unittest.main()
