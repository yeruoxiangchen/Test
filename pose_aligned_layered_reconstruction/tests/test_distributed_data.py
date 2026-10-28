from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pose_aligned_layered_reconstruction.background_round1 import atomic, require_unstarted_formal_run, sha
from pose_aligned_layered_reconstruction.distributed_data import verify_dataset_shard, validate_verification_receipts
from pose_aligned_layered_reconstruction.monitor_round1 import render


class FakeDataset:
    def __init__(self, root):
        self.manifest_path = root / "manifest.json"
        self.root = root
        self.rows = []
        for index in range(11):
            path = root / f"sample{index}.pt"
            path.write_bytes(bytes([index]) * 100)
            self.rows.append({"uid": str(index), "sample": path.name, "sha256": sha(path)})
        atomic(self.manifest_path, {"samples": self.rows})

    def __len__(self):
        return len(self.rows)

    def _resolve(self, path):
        return self.root / path


class DistributedDataTests(unittest.TestCase):
    def test_six_shards_cover_all_samples_once_and_reject_bad_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = FakeDataset(root)
            receipts = [verify_dataset_shard(dataset, rank, 6, root) for rank in range(6)]
            self.assertEqual(sum(r["verified_objects"] for r in receipts), 11)
            self.assertTrue(validate_verification_receipts(dataset, receipts)["all_sample_hashes_verified_before_training"])
            receipts[0]["verified_objects"] -= 1
            with self.assertRaises(ValueError):
                validate_verification_receipts(dataset, receipts)

    def test_corrupt_sample_still_fails_full_sha_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = FakeDataset(root)
            (root / "sample6.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                verify_dataset_shard(dataset, 0, 6, root)

    def test_only_unstarted_run_can_use_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic(root / "run_config.json", {"stage": "lari", "max_steps": 4000, "world_size": 6})
            require_unstarted_formal_run(root)
            (root / "history.jsonl").write_text('{"step":1}\n')
            with self.assertRaisesRegex(RuntimeError, "startup-only recovery forbidden"):
                require_unstarted_formal_run(root)

    def test_monitor_does_not_confuse_prepared_count_with_training_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render({"state": "running", "stage": "prepare_formal_data",
                             "progress": {"prepared_train2000_objects": 2000}}, Path(tmp))
            self.assertIn("总状态：运行中", result)
            self.assertIn("0 / 4000 updates；尚未开始", result)
            self.assertIn("数量齐不等于校验结束", result)


if __name__ == "__main__":
    unittest.main()
