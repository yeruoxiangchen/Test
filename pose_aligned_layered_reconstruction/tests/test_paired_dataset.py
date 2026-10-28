from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from pose_aligned_layered_reconstruction.paired_dataset import PairedLayerDataset, sha256_file, write_manifest
from pose_aligned_layered_reconstruction.tests.common import synthetic_sample


class DatasetTests(unittest.TestCase):
    def test_stage_a_does_not_require_trellis_context_and_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = synthetic_sample(channels=8)
            path = root / "sample.pt"
            torch.save(sample, path)
            manifest = root / "manifest.json"
            write_manifest(manifest, rows=[{
                "uid": sample["uid"], "object_uid": sample["object_uid"],
                "sample": path.name, "sha256": sha256_file(path),
            }], config={})
            loaded = PairedLayerDataset(manifest, build_trellis_contexts=False)[0]
            self.assertNotIn("stock_condition", loaded)
            sample["raw_visual_patch_features"] += 1
            torch.save(sample, path)
            with self.assertRaisesRegex(RuntimeError, "sample hash differs"):
                PairedLayerDataset(manifest, build_trellis_contexts=False)


if __name__ == "__main__":
    unittest.main()
