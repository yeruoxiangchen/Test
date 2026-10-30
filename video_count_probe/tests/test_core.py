import tempfile
import unittest
from pathlib import Path

from video_count_probe.core import (atomic_json, build_plan, checked_plan, make_case, file_sha, is_complete,
                                  prepare, token_groups, writer_lock)
from video_count_probe.evaluation import report, score_annotation, validate_annotation


class ExperimentTests(unittest.TestCase):
    def test_paired_design_and_same_relay_text(self):
        plan = build_plan("pilot")
        self.assertEqual(len(plan["jobs"]), 96)
        self.assertEqual(len(plan["cases"]), 12)
        pairs = {}
        for j in plan["jobs"]:
            pairs.setdefault((j["case_id"], j["seed"]), set()).add(j["method"])
        self.assertTrue(all(len(p) == 4 for p in pairs.values()))
        self.assertEqual(len({j['id'] for j in plan['jobs']}), 96)
        self.assertEqual(plan, build_plan("pilot"))

    def test_schedule_covers_latent_grid_and_full_cycles(self):
        for c in build_plan("pilot")["cases"]:
            self.assertEqual(c["segments"][0]["start"], 0)
            self.assertEqual(c["segments"][-1]["end"], 21)
            self.assertEqual(sum(s["kind"] == "action" for s in c["segments"]), c["target_count"])
            for previous, current in zip(c["segments"], c["segments"][1:]):
                self.assertEqual(previous["end"], current["start"])
            for s in c["segments"]:
                self.assertGreaterEqual(s["end"]-s["start"], 4)
                self.assertEqual(c["relay_prompt"][s["char_start"]:s["char_end"]], s["text"])
                self.assertLessEqual(s["end_seconds"], 5.0)

    def test_offsets_distinguish_repeated_words_and_padding(self):
        offsets = [(0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]
        segments = [{"start": 0, "end": 5, "char_start": 5, "char_end": 9},
                    {"start": 5, "end": 10, "char_start": 10, "char_end": 14}]
        self.assertEqual([g["tokens"] for g in token_groups(offsets, segments)], [[2], [3]])
        with self.assertRaises(ValueError):
            token_groups(offsets[:2], segments)

    def test_plan_tampering_and_duplicate_writer_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            prepare(temp, "pilot")
            p = Path(temp) / "pilot/plan.json"
            value = checked_plan(p)
            value["config"]["steps"] = 2
            atomic_json(p, value)
            with self.assertRaises(ValueError):
                checked_plan(p)
            with writer_lock(Path(temp) / "lock"):
                with self.assertRaises(RuntimeError):
                    with writer_lock(Path(temp) / "lock"):
                        pass

    def test_generated_count_is_not_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            prepare(temp, "pilot")
            result = report(Path(temp) / "pilot")
            self.assertFalse(result["metrics_ready"])
            self.assertEqual(result["missing"], 96)
            self.assertEqual(result["methods"], {})

    def test_resume_verifies_video_and_run_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = build_plan("smoke")["jobs"][0]
            identity = {"model": "pinned", "source": "current"}
            self.assertFalse(is_complete(root, job, identity))
            video = root / "clip.mp4"
            video.write_bytes(b"completed-output")
            record = {"status": "ok", "identity": identity, "job": job, "video": "clip.mp4", "video_sha256": file_sha(video)}
            atomic_json(root / "records" / (job["id"] + ".json"), record)
            self.assertTrue(is_complete(root, job, identity))
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                is_complete(root, job, {"model": "another"})
            video.write_bytes(b"incomplete")
            self.assertFalse(is_complete(root, job, identity))

    def test_complete_and_partial_pairs_never_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = prepare(temp, "pilot")
            run = Path(temp) / "pilot"
            identity = {"plan_hash": plan["plan_hash"]}
            atomic_json(run / "identity.json", identity)
            annotations = []
            # One complete paired condition, plus one unpaired Relay result.
            jobs = list(plan["jobs"][:4]) + [next(j for j in plan["jobs"][4:] if j["method"] == "relay")]
            for job in jobs:
                path = run / "videos" / (job["id"] + ".mp4")
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"fixture-video-identity")
                record = {"status": "ok", "job": job, "identity": identity, "video": str(path.relative_to(run)),
                          "video_sha256": file_sha(path), "initial_latent_sha256": str(job["seed"])}
                atomic_json(run / "records" / (job["id"] + ".json"), record)
                annotations.append({"job_id": job["id"], "video_sha256": file_sha(path), "rater": "test", "complete": True,
                                    "visually_usable": True, "action_present": True, "ambiguous": False, "completion_times": [2.0]})
            annotation_path = run / "annotations.json"
            atomic_json(annotation_path, {"plan_hash": plan["plan_hash"], "annotations": annotations})
            result = report(run, [annotation_path])
            self.assertFalse(result["metrics_ready"])
            self.assertEqual(result["generated"], 5)
            self.assertEqual(result["paired"]["relay_vs_relay_unrouted"]["pairs"], 1)
            with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
                report(run, [annotation_path, annotation_path])
            path.write_bytes(b"corrupted-video")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                report(run, [annotation_path])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan("pilot")
        self.case = make_case("clap", 3, "regular")
        self.record = {"job": {"id": "x"}, "video_sha256": "abc"}
        self.annotation = {"job_id": "x", "video_sha256": "abc", "rater": "human1", "complete": True,
                           "visually_usable": True, "action_present": True, "ambiguous": False,
                           "completion_times": [1.2, 2.8, 4.4]}

    def test_exact_usable_and_timing(self):
        validate_annotation(self.annotation, self.record, self.case, self.plan)
        self.assertEqual(score_annotation(self.annotation, self.case, self.plan)["exact_usable"], 1)
        self.annotation["completion_times"] = [0.2, 0.4, 0.6]
        scores = score_annotation(self.annotation, self.case, self.plan)
        self.assertEqual(scores["exact_usable"], 1)
        self.assertEqual(scores["timing_success"], 0)

    def test_unusable_does_not_disappear_from_primary_denominator(self):
        self.annotation["visually_usable"] = False
        scores = score_annotation(self.annotation, self.case, self.plan)
        self.assertEqual(scores["exact_usable"], 0)
        self.assertIsNone(scores["absolute_count_error"])

    def test_invalid_or_wrong_video_annotation_rejected(self):
        for change in ({"completion_times": [float("nan")]}, {"completion_times": [1, 1]},
                       {"completion_times": [9]}, {"video_sha256": "other"}, {"rater": ""},
                       {"action_present": False}, {"visually_usable": "false"}):
            annotation = dict(self.annotation, **change)
            with self.assertRaises(ValueError):
                validate_annotation(annotation, self.record, self.case, self.plan)


if __name__ == "__main__":
    unittest.main()

