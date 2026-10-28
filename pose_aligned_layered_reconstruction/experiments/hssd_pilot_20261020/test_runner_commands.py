"""CPU-only regressions for queue commands; never load models or launch jobs."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("hssd_runner_under_test", HERE / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerCommandsTest(unittest.TestCase):
    def test_evaluation_model_stage_is_a_cli_option(self):
        for stage in ("ss", "slat"):
            with self.subTest(stage=stage):
                argv = runner.command("evaluate", stage=stage, mode="parent", seeds=42)
                self.assertEqual(argv[3], runner.MODULE + ".evaluate")
                self.assertEqual(argv[argv.index("--stage") + 1], stage)
                self.assertEqual(argv[argv.index("--seeds") + 1], "42")

    def test_every_command_call_site_accepts_its_keywords(self):
        # Includes both the smoke and long-evaluation paths that previously
        # crashed before a subprocess was created.
        tree = ast.parse((HERE / "runner.py").read_text())
        checked = 0
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "command":
                module = ast.literal_eval(call.args[0])
                options = {kw.arg: "fixture" for kw in call.keywords}
                with self.subTest(module=module, options=tuple(options)):
                    argv = runner.command(module, **options)
                    self.assertEqual(argv[3], runner.MODULE + "." + module)
                checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_paired_evaluation_dispatches_both_modes_and_summary(self):
        for stage in ("ss", "slat"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / "evaluations" / "probe"
                folder.mkdir(parents=True)
                expected = {"fit_advantage_established": False}
                (folder / "summary.json").write_text(json.dumps(expected))
                with patch.object(runner, "OUT", root), patch.object(runner, "run_child") as child:
                    result = runner.evaluation("probe", stage, root / "manifest.json", root / "last.pt")
                self.assertEqual(result, expected)
                self.assertEqual(child.call_count, 3)
                for call, mode in zip(child.call_args_list[:2], ("parent", "candidate")):
                    name, argv = call.args
                    self.assertEqual(name, "probe_" + mode)
                    self.assertEqual(argv[argv.index("--stage") + 1], stage)
                    self.assertEqual(argv[argv.index("--mode") + 1], mode)
                self.assertIn("--summarize", child.call_args_list[-1].args[1])

    def test_completed_stage_is_reused_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "completed").mkdir()
            argv = ["python", "finished_stage.py"]
            (root / "completed" / "finished.json").write_text(json.dumps({"command": argv}))
            with patch.object(runner, "OUT", root), patch.object(runner.subprocess, "Popen") as launch:
                runner.run_child("finished", argv)
                launch.assert_not_called()

    def test_completed_smoke_is_verified_and_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            source = root / "source.py"
            source.write_text("# fixture\n")
            receipt = root / "smoke.json"
            receipt.write_text(json.dumps({"passed": True, "completed_at": "original-time",
                "sources": [runner.binding(source)], "manifest": runner.binding(manifest)}))
            before = receipt.read_bytes()
            with patch.object(runner, "OUT", root), patch.object(runner, "train") as train:
                runner.smoke()
                train.assert_not_called()
                self.assertEqual(receipt.read_bytes(), before)
                source.write_text("# changed\n")
                with self.assertRaisesRegex(ValueError, "Input changed"):
                    runner.smoke()


if __name__ == "__main__":
    unittest.main()
