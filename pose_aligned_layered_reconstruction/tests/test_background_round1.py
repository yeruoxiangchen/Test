import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from pose_aligned_layered_reconstruction.background_round1 import (
    PARENT_SHA, atomic, in_scope, sha, validate_records,
)
from pose_aligned_layered_reconstruction.paired_dataset import canonical_json_sha256
from pose_aligned_layered_reconstruction.prepare_official_experiment import read_selected_render_members


class BackgroundTests(unittest.TestCase):
    def test_stop_scope_rejects_other_projects_and_nonexperiment_jobs(self):
        module = "pose_aligned_layered_reconstruction.train_layered_ss"
        self.assertFalse(in_scope((7, "12", ("python", "-m", module, "--output_dir", "/tmp/test"))))
        self.assertFalse(in_scope((7, "12", ("python", "-m", "some.other.trainer",
            "--output_dir", "pose_aligned_layered_reconstruction/outputs/round1_20260909/test"))))
        self.assertTrue(in_scope((7, "12", ("python", "-m", module,
            "--output_dir", "pose_aligned_layered_reconstruction/outputs/round1_20260909/test"))))

    def make_evaluation(self, root):
        manifest = root / "manifest.json"
        atomic(manifest, {"samples": [{"uid": "abc"}]})
        output = root / "evaluation"
        identity = {"manifest_sha256": sha(manifest), "steps": 25, "cfg": 5, "seeds": "42,43",
            "parent": {"sha256": PARENT_SHA}, "candidate": {"step": 3000}, "layer_source": "predicted",
            "ground_truth_removed_from_predicted_condition": True}
        atomic(output / "identity_0_1.json", identity)
        for seed in (42, 43):
            for mode in ("parent", "candidate"):
                row = {"uid": "abc", "index": 0, "seed": seed, "mode": mode,
                       **dict.fromkeys(("iou", "precision", "recall", "f1", "count_ratio"), .5)}
                atomic(output / "records" / f"{seed}_{mode}.json", {
                    "identity_hash": canonical_json_sha256(identity), "records": [row]})
        return manifest, output

    def test_complete_evaluation_and_missing_seed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = self.make_evaluation(Path(tmp))
            _, rows = validate_records(output, manifest, ("parent", "candidate"), 3000)
            self.assertEqual(len(rows), 4)
            (output / "records/43_candidate.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_records(output, manifest, ("parent", "candidate"), 3000)

    def test_foreign_record_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = self.make_evaluation(Path(tmp))
            path = output / "records/42_candidate.json"
            row = json.loads(path.read_text())
            row["identity_hash"] = "wrong"
            atomic(path, row)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_records(output, manifest, ("parent", "candidate"), 3000)

    def test_selected_members_must_be_complete_but_unused_tail_not_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for name, payload in (("wanted", b"complete"), ("unused", b"x" * 2000)):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            path = Path(tmp) / "render.tar"
            path.write_bytes(buffer.getvalue()[:1700])
            self.assertEqual(read_selected_render_members(path, ["wanted"]), {"wanted": b"complete"})
            with self.assertRaises((tarfile.ReadError, ValueError)):
                read_selected_render_members(path, ["unused"])


if __name__ == "__main__":
    unittest.main()
