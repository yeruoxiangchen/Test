import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from screenshot_agent.config import Limits, RunConfig, ScreenshotPolicy, Task
from screenshot_agent.data import build_dataset
from screenshot_agent.report import edit_distance, make_report
from screenshot_agent.runner import run_task


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixture_data") / "dataset"
    report = build_dataset(directory, limit=2)
    assert all(v["baseline_returncode"] != 0 and v["reference_returncode"] == 0 for v in report["audit"])
    return directory


@pytest.mark.sandbox
@pytest.mark.parametrize("name", ["add", "mean"])
def test_recorded_end_to_end_is_explicitly_not_accuracy(dataset, tmp_path, name):
    task = Task.from_file(dataset / f"synthetic_{name}/task_light.json")
    config = RunConfig(limits=Limits(max_wall_seconds=120))
    result = run_task(task, config, tmp_path / "run", reader_name="oracle", model_name="replay")
    assert result["status"] == "verified", result
    assert result["resolved"]
    assert result["model_inference"] is False
    assert result["validation"]["baseline"]["returncode"] != 0
    assert result["validation"]["final"]["returncode"] == 0
    assert "reference patch" in result["interpretation"]
    assert (tmp_path / "run/patch.diff").stat().st_size > 0
    assert Path(task.repo, "miniapp.py").read_text() != (tmp_path / "run/workspace/miniapp.py").read_text()
    report = make_report(tmp_path)
    assert report["summary"][0]["code_backend"] == "replay"
    assert "scripted plumbing" in report["summary"][0]["interpretation"]


@pytest.mark.sandbox
def test_step_budget_stops_without_claiming_success(dataset, tmp_path):
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    result = run_task(task, RunConfig(limits=Limits(max_steps=1)), tmp_path / "limited",
                      reader_name="oracle", model_name="replay")
    assert not result["resolved"]
    assert result["exit_reason"] == "step_limit"


def test_missing_live_config_is_materialized_failure(dataset, tmp_path):
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    result = run_task(task, RunConfig(), tmp_path / "missing", reader_name="vlm", model_name="api")
    assert result["status"] == "error"
    assert result["failure_stage"] == "configuration"
    assert json.loads((tmp_path / "missing/result.json").read_text())["resolved"] is False


@pytest.mark.sandbox
def test_model_error_after_edit_preserves_patch_and_independent_verification(dataset, tmp_path, monkeypatch):
    class InterruptedAgent:
        def __init__(self, llm, tools, **kwargs):
            self.tools, self.messages = tools, []

        def chat(self, prompt):
            tool = next(t for t in self.tools if t.name == "edit_file")
            tool.execute(file_path="miniapp.py", old_string="return a - b", new_string="return a + b")
            raise RuntimeError("Simulated model context error after a source edit")

    monkeypatch.setattr("screenshot_agent.runner.Agent", InterruptedAgent)
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    result = run_task(task, RunConfig(), tmp_path / "interrupted", reader_name="oracle", model_name="replay")
    assert result["exit_reason"] == "model_error"
    assert result["resolved"] and result["validation"]["final"]["returncode"] == 0
    assert (tmp_path / "interrupted/patch.diff").stat().st_size > 0
    assert "Simulated model context error" in (tmp_path / "interrupted/agent_error.log").read_text()


def test_edit_distance_counts_whitespace():
    assert edit_distance("    return 1", "return 1") == 4
    assert edit_distance("abc", "axc") == 1


@pytest.mark.sandbox
def test_initial_only_policy_reads_once_and_hides_image_tools(dataset,tmp_path,monkeypatch):
    observed={}
    class InspectAgent:
        def __init__(self,llm,tools,**kwargs):
            observed['tools']={t.name for t in tools}
            self.messages=[]
        def chat(self,prompt):
            observed['prompt']=prompt
            return 'No patch submitted in this integration check.'
    monkeypatch.setattr('screenshot_agent.runner.Agent',InspectAgent)
    task=Task.from_file(dataset/'synthetic_add/task_light.json')
    config=RunConfig(screenshots=ScreenshotPolicy(allow_crop=False,allow_reread=False))
    result=run_task(task,config,tmp_path/'initial_only',reader_name='oracle',model_name='replay')
    assert result['usage']['screenshot_read_calls']==1
    assert 'return a - b' in observed['prompt']
    assert not {'read_screenshot','crop_screenshot'} & observed['tools']
    assert {'grep','edit_file','run_tests'} <= observed['tools']
    assert result['resolved'] is False


@pytest.mark.sandbox
def test_unreadable_image_materializes_failure_before_agent(dataset,tmp_path):
    task=Task.from_file(dataset/'synthetic_add/task_light.json')
    invalid=tmp_path/'broken.png'
    invalid.write_bytes(b'this is not a PNG image')
    task.images[0].path=str(invalid)
    result=run_task(task,RunConfig(),tmp_path/'broken',reader_name='oracle',model_name='replay')
    assert result['status']=='error' and result['failure_stage']=='reading'
    assert result['error_type']=='UnidentifiedImageError'
    assert not result['resolved']
    assert json.loads((tmp_path/'broken/result.json').read_text())['status']=='error'


@pytest.mark.sandbox
def test_direct_repair_cli_exports_verified_patch_without_mutating_input(dataset, tmp_path, monkeypatch):
    from openai.types.chat import ChatCompletion

    from screenshot_agent.cli import main
    from screenshot_agent.config import ModelConfig
    from screenshot_agent.models import APIModel

    monkeypatch.setenv("SCA_TEST_KEY", "scripted-integration-only")
    config = RunConfig(code=ModelConfig(model="scripted-test", api_key_env="SCA_TEST_KEY",
                        input_usd_per_million=0, output_usd_per_million=0))
    config_file = tmp_path / "config.json"
    config_file.write_text(config.model_dump_json())
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    source = Path(task.repo, "miniapp.py").read_text()
    messages = iter([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "edit", "type": "function", "function": {
            "name": "edit_file", "arguments": json.dumps({"file_path": "miniapp.py", "old_string": "return a - b",
                                                           "new_string": "return a + b"})}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "test", "type": "function", "function": {
            "name": "run_tests", "arguments": "{}"}}]},
        {"role": "assistant", "content": "Scripted integration check complete."},
    ])

    def complete(self, *args, **kwargs):
        return ChatCompletion.model_validate({"id": "scripted", "object": "chat.completion", "created": 0,
            "model": "scripted-test", "choices": [{"index": 0, "finish_reason": "stop", "message": next(messages)}]})

    monkeypatch.setattr(APIModel, "complete", complete)
    output = tmp_path / "runs"
    assert main(["repair", "--repo", task.repo, "--image", task.images[0].path,
                 "--problem", "Addition is incorrect", "--test-command", task.test_command,
                 "--reader", "ocr", "--config", str(config_file), "--output-root", str(output), "--run-id", "direct"]) == 0
    result = json.loads((output / "direct/result.json").read_text())
    assert result["validation"]["baseline"]["returncode"] != 0
    assert result["validation"]["final"]["returncode"] == 0
    assert result["changed_files"] == ["miniapp.py"]
    assert Path(task.repo, "miniapp.py").read_text() == source
    created = json.loads((output / "direct/task.json").read_text())
    assert created["reference_patch"] is None
    assert created["images"][0]["reference"] is None


@pytest.mark.sandbox
def test_ordinary_ocr_really_reads_image(dataset, tmp_path):
    from screenshot_agent.control import Events
    from screenshot_agent.images import ImageStore, OCRReader
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    store = ImageStore(task.images, tmp_path / "images", Events(tmp_path / "events.jsonl"))
    reading = OCRReader(store).read("shot")
    assert "return" in reading.raw_text.lower()
    assert "a" in reading.raw_text


@pytest.mark.sandbox
def test_baseline_reproduction_cannot_mutate_private_snapshot(dataset, tmp_path):
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    original = Path(task.repo, "miniapp.py").read_text()
    task.test_command = "python3 -c \"from pathlib import Path; Path('miniapp.py').write_text('changed'); raise ValueError('repro')\""
    result = run_task(task, RunConfig(limits=Limits(max_steps=1)), tmp_path / "side_effect",
                      reader_name="oracle", model_name="replay")
    assert not result["resolved"]
    assert (tmp_path / "side_effect/baseline/miniapp.py").read_text() == original
    assert (tmp_path / "side_effect/baseline_check/miniapp.py").read_text() == "changed"
    assert Path(task.repo, "miniapp.py").read_text() == original


@pytest.mark.sandbox
def test_background_status_and_stop_cancel_active_sandbox(dataset, tmp_path):
    from screenshot_agent.cli import main, start_ticks
    task = Task.from_file(dataset / "synthetic_add/task_light.json")
    task.test_command = "python3 -c 'import time; time.sleep(60); raise ValueError(\"repro\")'"
    task_file = tmp_path / "task.json"
    task_file.write_text(task.model_dump_json())
    root = tmp_path / "jobs"
    child = subprocess.run([sys.executable, "-m", "screenshot_agent", "run", "--task", str(task_file),
                            "--model", "replay", "--reader", "oracle", "--background", "--run-id", "cancel",
                            "--output-root", str(root)], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    launched = json.loads(child.stdout)
    directory, pid = Path(launched["run_dir"]), launched["pid"]
    try:
        for _ in range(150):
            path = directory / "status.json"
            if path.exists() and json.loads(path.read_text()).get("stage") == "baseline_test":
                break
            time.sleep(.1)
        else:
            pytest.fail((directory / "worker.log").read_text())
        assert main(["status", "--run-dir", str(directory)]) == 0
        assert main(["stop", "--run-dir", str(directory)]) == 0
        for _ in range(100):
            if (directory / "result.json").exists() and start_ticks(pid) is None:
                break
            time.sleep(.1)
        result = json.loads((directory / "result.json").read_text())
        assert result["status"] == "cancelled"
        assert not result["resolved"]
    finally:
        if start_ticks(pid) is not None:
            os.kill(pid, signal.SIGTERM)
