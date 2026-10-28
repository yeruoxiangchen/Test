import importlib.util
import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from screenshot_agent.config import RunConfig, RuntimeConfig, Task
from screenshot_agent.control import atomic_json
from screenshot_agent.repository import git, validate_tree
from screenshot_agent.sandbox import BubblewrapSandbox
from screenshot_agent.scale_data import (
    audit_one,
    balanced_order,
    choose_records,
    copy_image_repository,
    finalize,
    immutable_json,
    sha256,
    tree_hash,
)
from screenshot_agent.tool_probe import PROTOCOL, REQUIRED_CHECKS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scale_controller", ROOT / "scripts/run_scale_4gpu.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def rows(split):
    return [{"instance_id": f"{split}_{repo}_{i}", "repo": repo, "problem_statement": "No fenced code block."}
            for repo in ("a/repo", "b/repo", "c/repo") for i in range(8)]


def test_selection_is_balanced_excludes_seen_and_uses_original_splits():
    source = {s: rows(s) for s in ("dev", "test")}
    # Use path-safe instance IDs, independent of repository name.
    for items in source.values():
        for item in items:
            item["instance_id"] = item["instance_id"].replace("/", "__")
    excluded = {source["test"][0]["instance_id"]}
    result = choose_records(source, excluded, set(), {"dev": 9, "test": 12}, {"dev": 3, "test": 3}, 42)
    assert len(result) == 27
    assert not excluded.intersection(r["instance_id"] for r in result)
    assert all(r["sca_split"] == r["source_split"] for r in result)
    assert CounterLike(result, "dev") == {"a/repo": 4, "b/repo": 4, "c/repo": 4}
    assert result == choose_records(source, excluded, set(), {"dev": 9, "test": 12}, {"dev": 3, "test": 3}, 42)
    assert balanced_order(list(reversed(source["dev"])), 42) == balanced_order(source["dev"], 42)


def CounterLike(records, split):
    from collections import Counter
    return dict(Counter(r["repo"] for r in records if r["sca_split"] == split))


def test_duplicate_official_split_is_rejected():
    source = {s: [{"instance_id": "same", "repo": "a/b"}, {"instance_id": s, "repo": "a/b"}]
              for s in ("dev", "test")}
    with pytest.raises(ValueError, match="Duplicate"):
        choose_records(source, set(), set(), {"dev": 1, "test": 1}, {"dev": 1, "test": 1}, 42)


def test_patch_newline_normalization_is_recorded_without_changing_source():
    source = {s: [{"instance_id": f"{s}{i}", "repo": "a/b", "patch": "diff\r\nline\r\n"}
                  for i in range(2)] for s in ("dev", "test")}
    selected = choose_records(source, set(), set(), {"dev": 1, "test": 1}, {"dev": 1, "test": 1}, 42)
    assert all(r["patch"] == "diff\nline\n" and r["patch_crlf_normalized"] == ["patch"] for r in selected)
    assert source["dev"][0]["patch"] == "diff\r\nline\r\n"


def test_snapshot_materializes_only_internal_links_and_excludes_history(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".git/gold_secret").write_text("never export history")
    (source / "code.py").write_text("print(1)\n")
    (source / "alias.py").symlink_to("code.py")
    metadata = copy_image_repository(source, tmp_path / "export")
    assert (tmp_path / "export/alias.py").read_text() == "print(1)\n"
    assert not (tmp_path / "export/alias.py").is_symlink()
    assert not (tmp_path / "export/.git").exists()
    assert metadata["materialized_links"]
    (source / "outside").symlink_to(tmp_path / "unrelated")
    with pytest.raises(ValueError, match="leaves"):
        copy_image_repository(source, tmp_path / "bad")


def test_tree_fingerprint_and_explicit_limits(tmp_path):
    (tmp_path / "x").write_text("12345")
    before = tree_hash(tmp_path)
    (tmp_path / "x").write_text("12346")
    assert tree_hash(tmp_path) != before
    with pytest.raises(ValueError, match="exceeds"):
        validate_tree(tmp_path, max_bytes=4)
    assert RuntimeConfig().max_repo_bytes == 100_000_000


def test_partial_preflight_cannot_create_formal_manifests(tmp_path):
    records = [{"instance_id": f"{s}{i}", "repo": "a/b", "sca_split": s, "in_verified": False}
               for s in ("dev", "test") for i in range(3)]
    atomic_json(tmp_path / "selection.json", {"targets": {"dev": 2, "test": 2}})
    atomic_json(tmp_path / "selected_records.json", records)
    for row in records[:5]:
        atomic_json(tmp_path / "tasks" / row["instance_id"] / "audit.json", {
            "instance_id": row["instance_id"], "status": "environment_failed" if row["instance_id"] == "dev0" else "validated"})
    assert not finalize(tmp_path)["ready"]
    assert not (tmp_path / "dev.jsonl").exists()
    atomic_json(tmp_path / "tasks/test2/audit.json", {"instance_id": "test2", "status": "validated"})
    report = finalize(tmp_path)
    assert report["ready"]
    assert report["preflight_states"]["environment_failed"] == 1
    assert [r["task_id"] for r in controller.jsonlines(tmp_path / "dev.jsonl")] == ["dev1", "dev2"]
    # Retrying a failed primary must not silently change an already fixed evaluation manifest.
    atomic_json(tmp_path / "tasks/dev0/audit.json", {"instance_id": "dev0", "status": "validated"})
    with pytest.raises(ValueError, match="already frozen"):
        finalize(tmp_path)


def test_frozen_inputs_are_not_overwritten(tmp_path):
    path = tmp_path / "selection.json"
    immutable_json(path, {"seed": 42})
    immutable_json(path, {"seed": 42})
    with pytest.raises(ValueError, match="Frozen"):
        immutable_json(path, {"seed": 43})


@pytest.mark.parametrize("gpus", ["6", "4,5,6", "4,4,6,7", "a,5,6,7", "0,1,2,3,4"])
def test_coder_refuses_non_four_gpu_configuration(gpus):
    with pytest.raises(ValueError):
        controller.parse_gpus(gpus)


def test_coder_command_uses_tp4_and_pinned_native_tool_parser(tmp_path):
    command = controller.server_command(tmp_path, "coder", "4,5,6,7", 18836, {"tool_call_parser": "qwen3_coder"})
    assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.8"
    assert "--trust-remote-code" not in command
    assert "--api-key" not in command  # supplied in environment, not process arguments/logs


def test_signal_tracing_preserves_model_command_and_records_only_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.shutil, "which", lambda name: "/usr/bin/strace")
    original = controller.server_command(tmp_path, "coder", "2,5,6,7", 18836,
                                         {"tool_call_parser": "qwen3_coder"})
    command = controller.signal_trace_command(original, tmp_path / "signals.log")
    assert command[0] == "/usr/bin/strace"
    assert command[command.index("--") + 1:] == original
    assert "-DDD" in command and "-f" in command and "--seccomp-bpf" in command
    assert "trace=none" in command and "signal=SIGTERM,SIGINT,SIGHUP,SIGKILL" in command
    assert command[command.index("-o") + 1] == str(tmp_path / "signals.log")


def test_signal_tracing_fails_before_gpu_checks_when_strace_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.shutil, "which", lambda name: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No GPU check or launch is allowed without requested tracer")
    monkeypatch.setattr(controller, "wait_for_gpu_capacity", forbidden)
    args = SimpleNamespace(gpus="2,5,6,7", port=18836, split="dev", gpu_memory_utilization=0.8,
                           trace_server_signals=True)
    with pytest.raises(ValueError, match="requires strace"):
        controller.start_service(SimpleNamespace(output=tmp_path, launch=forbidden), args,
                                 tmp_path, "baseline", {})


def test_signal_trace_parses_latest_sender_and_does_not_read_sender_process(tmp_path):
    path = tmp_path / "signals.log"
    path.write_text(
        '123 1792900000.000001 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=7, si_uid=1011} ---\n'
        '123 1792900000.000002 --- SIGINT {si_signo=SIGINT, si_code=SI_USER, si_pid=8, si_uid=1011} ---\n'
        '123 1792900000.000003 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=9, si_uid=0} ---\n'
        '123 1792900000.000004 +++ killed by SIGTERM +++\n')
    result = controller.signal_trace_summary({"signal_trace_log": str(path)})
    assert result["status"] == "sender_recorded"
    assert result["last_sigterm"]["sender_pid"] == 9 and result["last_sigterm"]["sender_uid"] == 0
    assert result["last_sigterm"]["si_code"] == "SI_USER"
    assert "si_pid=9" in result["last_sigterm"]["line"]


def test_signal_trace_unknown_latest_sender_cannot_reuse_older_sender(tmp_path):
    path = tmp_path / "signals.log"
    path.write_text(
        '123 1.1 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=7, si_uid=1011} ---\n'
        '123 1.2 --- SIGTERM {si_signo=SIGTERM, si_code=SI_KERNEL} ---\n')
    result = controller.signal_trace_summary({"signal_trace_log": str(path)})
    assert result["status"] == "sender_unknown"
    assert result["last_sigterm"]["sender_pid"] is None
    assert result["last_sigterm"]["sender_uid"] is None


def test_signal_trace_prefers_model_recipient_over_later_worker_signal(tmp_path):
    path = tmp_path / "signals.log"
    path.write_text(
        '123 1.1 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=7, si_uid=1011} ---\n'
        '124 1.2 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=8, si_uid=1011} ---\n')
    direct = controller.signal_trace_summary({"pid": 123, "signal_trace_log": str(path)})["last_sigterm"]
    assert direct["recipient_pid"] == 123 and direct["sender_pid"] == 7
    assert direct["scope"] == "model_process"
    child_only = controller.signal_trace_summary({"pid": 999, "signal_trace_log": str(path)})["last_sigterm"]
    assert child_only["recipient_pid"] == 124 and child_only["sender_pid"] == 8
    assert child_only["scope"] == "observed_tracee"


def test_server_exit_receipt_includes_observed_sender_without_sender_process_reads(tmp_path):
    path = tmp_path / "signals.log"
    path.write_text('123 1.1 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=7, si_uid=1011} ---\n')
    updates = []
    c = SimpleNamespace(output=tmp_path, update=lambda **value: updates.append(value))
    service = {"pid": 123, "start_ticks": "fixture", "log": "server.log", "signal_trace_log": str(path)}
    result = controller.record_server_exit(c, SimpleNamespace(split="dev"), "baseline", service, [], -15)
    assert result["signal_trace"]["last_sigterm"]["sender_pid"] == 7
    assert result["signal_trace"]["last_sigterm"]["scope"] == "model_process"
    assert updates[-1]["model_server_exit"]["signal_trace"] == result["signal_trace"]


def test_signal_trace_cli_fails_before_creating_run_when_strace_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.shutil, "which", lambda name: None)
    output = tmp_path / "unstarted"
    monkeypatch.setattr(sys, "argv", ["run_scale_4gpu.py", "run", "--output", str(output),
                                     "--trace-server-signals"])
    with pytest.raises(SystemExit) as failure:
        controller.main()
    assert failure.value.code == 2
    assert not output.exists()


def test_signal_trace_absent_or_bounded_tail_is_explicitly_unknown(tmp_path):
    assert controller.signal_trace_summary({})["status"] == "disabled"
    path = tmp_path / "signals.log"
    assert controller.signal_trace_summary({"signal_trace_log": str(path)})["status"] == "unavailable"
    path.write_text('123 1.1 --- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=7, si_uid=1011} ---\n'
                    + 'unrelated trace line\n' * 10000)
    result = controller.signal_trace_summary({"signal_trace_log": str(path)})
    assert result["status"] == "no_sigterm_observed" and result["last_sigterm"] is None


def test_shared_gpu_capacity_checks_each_rank_and_preserves_headroom():
    policy = {"allow_shared_gpus": True, "gpu_memory_utilization": 0.8, "headroom_mib": 1536}
    raw = "2, 24576, 1024, 23104\n4, 24576, 1940, 22188\n5, 24576, 1000, 23128\n7, 24576, 1024, 23104\n"
    report = controller.assess_gpu_memory(raw, ["2", "4", "5", "7"], policy)
    assert report["ready"]
    assert report["devices"][1]["required_free_mib"] == 21197
    # Three mostly empty GPUs cannot compensate for one rank lacking headroom.
    assert not controller.assess_gpu_memory(raw.replace("1940, 22188", "3400, 20728"),
                                           ["2", "4", "5", "7"], policy)["ready"]
    assert not controller.assess_gpu_memory(raw, ["2", "4", "5", "7"],
                                           {**policy, "allow_shared_gpus": False})["ready"]


def test_missing_gpu_telemetry_cannot_pass_capacity_check():
    policy = {"allow_shared_gpus": True, "gpu_memory_utilization": 0.8, "headroom_mib": 1536}
    with pytest.raises(ValueError, match="does not match"):
        controller.assess_gpu_memory("2, 24576, 1024, 23104\n", ["2", "4", "5", "7"], policy)


def test_waits_for_two_stable_capacity_samples_without_launching_or_killing(tmp_path, monkeypatch):
    args = SimpleNamespace(allow_shared_gpus=True, gpu_memory_utilization=0.8, gpu_headroom_mib=1536,
                           wait_for_gpus=True)
    states, sleeps = [], []
    c = SimpleNamespace(output=tmp_path, update=lambda **value: states.append(value))
    readings = iter(["2, 24576, 6000, 18128", "2, 24576, 1000, 23128", "2, 24576, 1100, 23028"])
    monkeypatch.setattr(controller.subprocess, "check_output", lambda *a, **k: next(readings))
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: sleeps.append(seconds))
    controller.wait_for_gpu_capacity(c, args, ["2"], "baseline")
    assert [s["stage"] for s in states] == ["waiting_for_gpu_memory", "waiting_for_gpu_memory", "gpu_memory_ready"]
    assert sleeps == [15, 15]
    assert json.loads((tmp_path / "gpu_checks.jsonl").read_text())["devices"][0]["free_mib"] == 23028


def test_insufficient_capacity_fails_without_wait_flag(tmp_path, monkeypatch):
    args = SimpleNamespace(allow_shared_gpus=True, gpu_memory_utilization=0.8, gpu_headroom_mib=1536,
                           wait_for_gpus=False)
    c = SimpleNamespace(output=tmp_path, update=lambda **value: None)
    monkeypatch.setattr(controller.subprocess, "check_output", lambda *a, **k: "2, 24576, 8000, 16128")
    with pytest.raises(ValueError, match="Insufficient free GPU memory"):
        controller.wait_for_gpu_capacity(c, args, ["2"], "baseline")
    assert not (tmp_path / "gpu_checks.jsonl").exists()


def test_controller_cleans_its_child_and_rejects_concurrent_controller(tmp_path):
    c = controller.Controller(tmp_path, "unit_test")
    try:
        entry = c.launch([sys.executable, "-c", "import time; time.sleep(30)"], "owned")
        with pytest.raises(BlockingIOError):
            controller.Controller(tmp_path, "second")
    finally:
        c.close()
    assert entry["proc"].poll() is not None


def test_report_preserves_failures_and_requires_both_arms(tmp_path):
    folder = tmp_path / "dev"
    tasks = [{"task_id": t} for t in ("one", "two", "three")]
    atomic_json(folder / "plan.json", {"tasks": tasks, "interpretation": "unit test"})
    for arm in ("baseline", "coder"):
        for i, task in enumerate(tasks):
            atomic_json(folder / arm / "runs" / task["task_id"] / "result.json", {
                "task_id": task["task_id"], "resolved": i == 0 or (arm == "coder" and i == 1),
                "model_inference": True, "group_id": "a/b", "status": "unresolved", "elapsed_seconds": 2,
                "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    value = controller.comparison(tmp_path, "dev")
    assert value["paired"] == {"retained": 1, "repaired": 1, "both_failed": 1}
    assert value["arms"]["coder"]["attempted"] == 3
    (folder / "coder/runs/three/result.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        controller.comparison(tmp_path, "dev")


@pytest.mark.parametrize("interruption", [
    {"status": "cancelled"},
    {"status": "unresolved", "exit_reason": "model_error", "agent_error_type": "APIConnectionError"},
])
def test_resume_keeps_completed_failures_and_archives_only_interruption(tmp_path, monkeypatch, interruption):
    folder = tmp_path / "dev/coder"
    run_dir = folder / "runs/task"
    config = RunConfig().model_dump()
    config["code"].update(model="sca-coder", base_url="http://127.0.0.1:18836/v1")
    config["vision"].update(model="unused", base_url="http://127.0.0.1:1/v1")
    atomic_json(run_dir / "config.json", config)
    result = {"task_id": "task", "model_inference": True, "status": "error", "resolved": False}
    atomic_json(run_dir / "result.json", result)
    args = SimpleNamespace(output=tmp_path, split="dev", port=18836, resume_interrupted=False)
    plan = {"runtime": str(tmp_path / "runtime"), "common_config": config,
            "tasks": [{"task_id": "task"}], "resources": {}}
    def would_start(*a, **k):
        raise RuntimeError("would_start_gpu")
    monkeypatch.setattr(controller, "start_service", would_start)
    controller.run_arm(None, args, plan, "coder")
    assert json.loads((run_dir / "result.json").read_text())["status"] == "error"
    atomic_json(run_dir / "result.json", {**result, **interruption})
    with pytest.raises(ValueError, match="resume-interrupted"):
        controller.run_arm(None, args, plan, "coder")
    args.resume_interrupted = True
    with pytest.raises(RuntimeError, match="would_start_gpu"):
        controller.run_arm(None, args, plan, "coder")
    assert not run_dir.exists()
    assert len(list((tmp_path / "interrupted_attempts/dev/coder").glob("*/result.json"))) == 1


@pytest.mark.parametrize("details", [
    {"exit_reason": "malformed_tool_output"},
    {"exit_reason": "model_error", "agent_error_type": "BadRequestError"},
    {"exit_reason": "model_error", "agent_error_type": "ValueError"},
    {"exit_reason": "model_error", "agent_error_type": "APITimeoutError"},
])
def test_model_failures_are_not_reclassified_as_interruption(details):
    assert not controller.interrupted_result({"status": "unresolved", **details})


def test_status_separates_interrupted_tasks_from_completed_failures(tmp_path):
    rows = [
        {"task_id": "malformed", "status": "unresolved", "exit_reason": "malformed_tool_output"},
        {"task_id": "cancelled", "status": "cancelled"},
        {"task_id": "connection", "status": "unresolved", "exit_reason": "model_error",
         "agent_error_type": "APIConnectionError"},
    ]
    for row in rows:
        atomic_json(tmp_path / "dev/baseline/runs" / row["task_id"] / "result.json", row)
    state = controller.inspect_status(tmp_path, tmp_path / "missing_data")
    assert state["evaluations"]["dev/baseline"] == {
        "completed": 1, "resolved": 0, "interrupted": 2,
        "interrupted_task_ids": ["cancelled", "connection"],
    }


@pytest.mark.parametrize("details", [
    {"status": "cancelled"},
    {"status": "unresolved", "exit_reason": "model_error", "agent_error_type": "APIConnectionError"},
])
def test_comparison_rejects_interrupted_results(tmp_path, details):
    atomic_json(tmp_path / "dev/plan.json", {"tasks": [{"task_id": "task"}]})
    atomic_json(tmp_path / "dev/baseline/runs/task/result.json", {
        "task_id": "task", "model_inference": True, **details,
    })
    with pytest.raises(ValueError, match="baseline incomplete: 0/1"):
        controller.comparison(tmp_path, "dev")
    assert not (tmp_path / "dev/comparison.json").exists()


def test_server_exit_receipt_is_saved_before_cleanup_with_active_tasks(tmp_path, monkeypatch):
    (tmp_path / "dev").mkdir()
    args = SimpleNamespace(output=tmp_path, split="dev", port=18836,
                           resume_interrupted=False, workers=1)
    plan = {"runtime": str(tmp_path / "runtime"), "common_config": RunConfig().model_dump(),
            "tasks": [{"task_id": "task", "task_file": "task.json"}], "resources": {}}
    updates = []
    worker = {"proc": SimpleNamespace(poll=lambda: None), "log": "worker.log"}
    c = SimpleNamespace(output=tmp_path, update=lambda **value: updates.append(value),
                        launch=lambda *a, **k: worker)
    polls = iter([None, -9])
    proc = SimpleNamespace(poll=lambda: next(polls))
    service = {"proc": proc, "pid": 123, "start_ticks": "456", "log": "server.log"}
    monkeypatch.setattr(controller, "start_service", lambda *a, **k: (service, {}))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    cleaned = []
    def cleanup(process):
        receipts = list((tmp_path / "logs").glob("dev_coder_server_exit_*.json"))
        assert len(receipts) == 1  # Must survive before terminating remaining children.
        receipt = json.loads(receipts[0].read_text())
        assert receipt["returncode"] == -9 and receipt["signal"] == 9
        assert receipt["active_task_ids"] == ["task"]
        assert receipt["log"] == "server.log" and receipt["pid"] == 123
        assert updates[-1]["model_server_exit"]["receipt"] == str(receipts[0])
        cleaned.append(process)
    monkeypatch.setattr(controller, "terminate", cleanup)
    with pytest.raises(RuntimeError, match="returncode=-9, signal=9"):
        controller.run_arm(c, args, plan, "coder")
    assert cleaned == [proc]


def test_startup_probe_waits_with_heartbeat_and_sends_only_one_post(tmp_path):
    release = threading.Event()
    calls, updates = [], []

    def open_request(request, timeout):
        calls.append((request, timeout))
        assert release.wait(5)
        return io.StringIO('{"ready": true}')

    def update(**value):
        updates.append(value)
        if value.get("protocol_smoke_seconds", 0) >= 1:
            release.set()

    c = SimpleNamespace(update=update)
    entry = {"proc": SimpleNamespace(poll=lambda: None), "log": "server.log"}
    try:
        result = controller.startup_chat(c, entry, SimpleNamespace(open=open_request),
                                         "one-request", "fixture", "tool_call", timeout=4)
    finally:
        release.set()
    assert result == {"ready": True}
    assert calls == [("one-request", 4)]
    assert len(updates) >= 3
    assert all(update["stage"] == "protocol_smoke" for update in updates)
    assert any("startup_progress" in update for update in updates)


def test_startup_probe_timeout_does_not_queue_another_request():
    release = threading.Event()
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        assert release.wait(5)
        return io.StringIO('{}')

    entry = {"proc": SimpleNamespace(poll=lambda: None), "log": "server.log"}
    try:
        with pytest.raises(TimeoutError, match="tool_call exceeded 0.05s.*no retry was queued"):
            controller.startup_chat(SimpleNamespace(update=lambda **k: None), entry,
                                    SimpleNamespace(open=open_request), "once", "fixture", "tool_call", timeout=0.05)
    finally:
        release.set()
    assert calls == ["once"]


def test_startup_probe_detects_server_exit_without_waiting_for_http_timeout():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        entered.set()
        assert release.wait(5)
        return io.StringIO('{}')

    entry = {"proc": SimpleNamespace(poll=lambda: -15 if entered.is_set() else None), "log": "server.log"}
    try:
        with pytest.raises(RuntimeError, match="startup protocol smoke tool_response.*returncode=-15"):
            controller.startup_chat(SimpleNamespace(update=lambda **k: None), entry,
                                    SimpleNamespace(open=open_request), "once", "fixture", "tool_response", timeout=30)
    finally:
        release.set()
    assert calls == ["once"]


@pytest.mark.parametrize("invalid", [None, "name", "arguments", "truncated_response"])
def test_startup_protocol_preserves_strict_validation_and_generation_budget(tmp_path, monkeypatch, invalid):
    first = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
        "id": "probe", "type": "function", "function": {
            "name": "wrong" if invalid == "name" else "echo_probe",
            "arguments": json.dumps({"text": "wrong" if invalid == "arguments" else "hello"}),
        }}]}}]}
    second = {"choices": [{"message": {"role": "assistant", "content": "hello"},
                           "finish_reason": "length" if invalid == "truncated_response" else "stop"}]}
    posts, updates = [], []

    def open_request(request, timeout):
        if request.full_url.endswith("/health"):
            response = io.StringIO('{}')
            response.status = 200
            return response
        posts.append((json.loads(request.data), timeout))
        return io.StringIO(json.dumps(first if len(posts) == 1 else second))

    monkeypatch.setattr(controller.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_request))
    c = SimpleNamespace(output=tmp_path, update=lambda **value: updates.append(value))
    entry = {"proc": SimpleNamespace(poll=lambda: None), "log": "server.log"}
    if invalid:
        with pytest.raises(ValueError):
            controller.ready_server(c, entry, 18836, "private-key", "fixture")
        assert not (tmp_path / "logs/fixture_protocol_smoke.json").exists()
    else:
        controller.ready_server(c, entry, 18836, "private-key", "fixture")
        assert (tmp_path / "logs/fixture_protocol_smoke.json").exists()
        assert updates[-1]["stage"] == "protocol_smoke_passed"
    assert len(posts) == (1 if invalid in {"name", "arguments"} else 2)
    assert all(timeout == 600 and payload["max_tokens"] == 256 and payload["temperature"] == 0
               for payload, timeout in posts)
    assert "private-key" not in json.dumps(updates)


@pytest.mark.parametrize("trace_signals", [False, True])
def test_startup_timeout_receipt_precedes_our_cleanup_signal(tmp_path, monkeypatch, trace_signals):
    state = {"stage": "protocol_smoke", "protocol_smoke_step": "tool_call",
             "protocol_smoke_seconds": 600, "protocol_smoke_timeout_seconds": 600}
    proc = SimpleNamespace(poll=lambda: None)
    service = {"proc": proc, "pid": 99999999, "start_ticks": "fixture", "log": "server.log"}
    commands = []
    def launch(command, *args, **kwargs):
        commands.append(command)
        return service
    c = SimpleNamespace(output=tmp_path, state=state, update=lambda **v: state.update(v), launch=launch)
    args = SimpleNamespace(gpus="2,5,6,7", port=18836, split="dev", gpu_memory_utilization=0.8,
                           trace_server_signals=trace_signals)
    monkeypatch.setattr(controller.shutil, "which", lambda name: "/usr/bin/strace")
    monkeypatch.setattr(controller, "wait_for_gpu_capacity", lambda *a: None)
    fake_socket = SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: None, bind=lambda *a: None)
    class Socket:
        def __enter__(self):
            return fake_socket
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(controller.socket, "socket", lambda: Socket())
    def fail(*a, **k):
        raise TimeoutError("startup protocol timed out")
    monkeypatch.setattr(controller, "ready_server", fail)
    cleaned = []
    def cleanup(process):
        receipts = list((tmp_path / "logs").glob("dev_baseline_startup_failure_*.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["phase"] == "protocol_smoke" and receipt["error_type"] == "TimeoutError"
        assert receipt["server_returncode_before_cleanup"] is None
        assert receipt["server_alive_before_cleanup"] and receipt["controller_cleanup_requested"]
        assert receipt["signal_trace"]["enabled"] == trace_signals
        assert state["model_signal_trace"]["enabled"] == trace_signals
        assert state["model_signal_trace"]["log"] == service.get("signal_trace_log")
        assert state["model_startup_failure"]["receipt"] == str(receipts[0])
        cleaned.append(process)
    monkeypatch.setattr(controller, "terminate", cleanup)
    with pytest.raises(TimeoutError, match="startup protocol timed out"):
        controller.start_service(c, args, tmp_path, "baseline", {})
    assert cleaned == [proc]
    assert len(commands) == 1
    assert (commands[0][0] == "/usr/bin/strace") == trace_signals


def test_preflight_dev_retries_failures_without_touching_success_or_test(tmp_path, monkeypatch):
    data = tmp_path / "data"
    records = [{"instance_id": name, "sca_split": split} for name, split in
               [("d_ok", "dev"), ("d_failed", "dev"), ("d_pending", "dev"), ("t_pending", "test")]]
    atomic_json(data / "selection.json", {"upstream_files_sha256": {}})
    atomic_json(data / "selected_records.json", records)
    atomic_json(data / "tasks/d_ok/audit.json", {"status": "validated", "preserve": True})
    atomic_json(data / "tasks/d_failed/audit.json", {"status": "error"})
    original = (data / "tasks/d_ok/audit.json").read_bytes()
    calls = []
    def launch(command, label):
        name = command[command.index("--instance") + 1]
        calls.append(name)
        assert "--retry-failed" in command
        atomic_json(data / "tasks" / name / "audit.json", {"status": "validated"})
        return {"proc": SimpleNamespace(poll=lambda: 0, returncode=0), "log": "fixture.log"}
    finalized = []
    def finish(folder, split=None):
        finalized.append(split)
        return {"ready": False, "ready_by_split": {"dev": True, "test": False}, "audits": []}
    monkeypatch.setattr(controller, "finalize", finish)
    c = SimpleNamespace(launch=launch, update=lambda **kw: None)
    args = SimpleNamespace(data=data, output=tmp_path / "output", preflight_split="dev", limit=0,
                           retry_failed=True, workers=2, test_timeout=10)
    controller.preflight(c, args)
    assert calls == ["d_failed", "d_pending"]
    assert finalized == ["dev"]
    assert not (data / "tasks/t_pending").exists()
    assert (data / "tasks/d_ok/audit.json").read_bytes() == original


def test_preflight_seal_protects_only_the_evaluated_split(tmp_path, monkeypatch):
    data = tmp_path / "data"
    atomic_json(data / "selected_records.json", [])
    atomic_json(data / "selection.json", {"upstream_files_sha256": {}})
    atomic_json(data / "evaluation_seal_dev.json", {"split": "dev"})
    monkeypatch.setattr(controller, "finalize", lambda folder, split=None: {
        "ready_by_split": {"dev": True, "test": True}, "audits": []})
    c = SimpleNamespace(update=lambda **kw: None)
    args = SimpleNamespace(data=data, output=tmp_path / "output", preflight_split="test", limit=0,
                           retry_failed=True, workers=2, test_timeout=10)
    controller.preflight(c, args)  # A dev seal cannot block preparation of the held-out split.
    args.preflight_split = "dev"
    with pytest.raises(ValueError, match="already uses this split"):
        controller.preflight(c, args)
    args.preflight_split = "all"
    with pytest.raises(ValueError, match="already uses this split"):
        controller.preflight(c, args)


@pytest.mark.parametrize("agent_mode", ["v6", "v7"])
def test_dev_plan_keeps_exact_target_and_does_not_need_test_manifest(tmp_path, monkeypatch, agent_mode):
    data, output, runtime = tmp_path / "data", tmp_path / "output", tmp_path / "runtime"
    output.mkdir()
    task_dir = data / "tasks/d0"
    (task_dir / "repo").mkdir(parents=True)
    (task_dir / "repo/value.py").write_text("value = 0\n")
    for filename in ("task.json", "benchmark.json", "reference.patch"):
        (task_dir / filename).write_text("{}")
    audit = {"status": "validated", "repo_sha256": tree_hash(task_dir / "repo")}
    for name, key in (("task.json", "task_sha256"), ("benchmark.json", "benchmark_sha256"),
                      ("reference.patch", "reference_sha256")):
        audit[key] = sha256(task_dir / name)
    atomic_json(task_dir / "audit.json", audit)
    atomic_json(data / "selection.json", {"targets": {"dev": 1, "test": 1}, "selected_ids": ["d0", "t0"]})
    atomic_json(data / "preflight_report.json", {"ready": False, "ready_by_split": {"dev": True, "test": False}})
    row = {"task_id": "d0", "split": "dev", "task_file": str(task_dir / "task.json"),
           "audit_sha256": sha256(task_dir / "audit.json")}
    (data / "dev.jsonl").write_text(json.dumps(row) + "\n")
    pins = {"coder": {"revision": "fixture"}, "vllm_version": "fixture"}
    atomic_json(runtime / "configs/scale-4gpu.json", RunConfig().model_dump())
    atomic_json(runtime / "configs/scale_4gpu_resources.json", pins)
    atomic_json(runtime / "source_state.json", {"sha256": {}})
    atomic_json(tmp_path / ".coder-env/setup_complete.json", {"vllm_version": "fixture"})
    atomic_json(tmp_path / "model/download_complete.json", {"revision": "fixture"})
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "MODEL", tmp_path / "model")
    monkeypatch.setattr(controller, "frozen_runtime", lambda folder: runtime)
    args = SimpleNamespace(data=data, output=output, split="dev", allow_shared_gpus=True,
                           gpu_memory_utilization=0.8, gpu_headroom_mib=1536, agent_mode=agent_mode)
    updates = []
    plan = controller.plan_evaluation(args, progress=lambda **fields: updates.append(fields))
    assert len(plan["tasks"]) == 1
    assert updates[0]["validation_task"] == "d0" and updates[0]["validation_completed"] == 0
    assert updates[-1]["validation_completed"] == updates[0]["validation_total"] == 1
    assert plan["common_config"]["agent"]["mode"] == agent_mode
    assert plan["tool_preflight_required"] is True
    assert f"fixed {agent_mode} harness" in plan["interpretation"]
    assert (data / "evaluation_seal_dev.json").exists()
    assert not (data / "test.jsonl").exists()
    assert not (data / "evaluation_seal_test.json").exists()
    args.agent_mode = "v7" if agent_mode == "v6" else "v6"
    with pytest.raises(ValueError, match="Frozen"):
        controller.plan_evaluation(args)  # A resumed output cannot silently switch harness mode.
    args.agent_mode = agent_mode
    (data / "dev.jsonl").write_text("")
    with pytest.raises(ValueError, match="exactly 1"):
        controller.plan_evaluation(args)
    args.split = "test"
    with pytest.raises(ValueError, match="full declared test target"):
        controller.plan_evaluation(args)


def test_resume_reuses_resources_and_reaches_dev_before_test_preflight(tmp_path, monkeypatch):
    output, data = tmp_path / "output", tmp_path / "data"
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not install, download or run a separate smoke")
    def prepared(c, args):
        calls.append("preflight:" + args.preflight_split)
        atomic_json(data / "preflight_report.json", {"ready_by_split": {"dev": True, "test": False}})
    monkeypatch.setattr(controller, "resources", forbidden)
    monkeypatch.setattr(controller, "smoke", forbidden)
    monkeypatch.setattr(controller, "verify_resource_receipts", lambda: calls.append("receipts") or {})
    monkeypatch.setattr(controller, "preflight", prepared)
    monkeypatch.setattr(controller, "plan_evaluation", lambda args, **kw: calls.append("plan") or {})
    monkeypatch.setattr(controller, "preflight_tools", lambda c, args, plan: calls.append("tool_preflight"))
    monkeypatch.setattr(controller, "run_arm", lambda c, args, plan, arm: calls.append("run:" + arm))
    monkeypatch.setattr(controller, "comparison", lambda output, split: calls.append("comparison"))
    monkeypatch.setattr(controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["run_scale_4gpu.py", "resume", "--output", str(output), "--data", str(data),
                                     "--gpus", "2,4,5,7", "--allow-shared-gpus", "--retry-failed"])
    controller.main()
    assert calls == ["receipts", "preflight:dev", "plan", "tool_preflight", "run:baseline", "run:coder", "comparison"]
    assert json.loads((output / "status.json").read_text())["stage"] == "completed"
    calls.clear()
    controller.main()
    assert calls == ["receipts", "plan", "tool_preflight", "run:baseline", "run:coder", "comparison"]


def tool_preflight_fixture(tmp_path):
    runtime = tmp_path / "runtime"
    worker = runtime / "src/screenshot_agent/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# frozen worker fixture\n")
    worker.with_name("tool_probe.py").write_text("# filesystem-tools contract fixture\n")
    atomic_json(runtime / "source_state.json", {"sha256": {"src/screenshot_agent/worker.py": sha256(worker)}})
    task_path = tmp_path / "data/tasks/task0/task.json"
    atomic_json(task_path, {"task_id": "task0", "repo": "repo", "problem": "fix",
                            "runtime": {"backend": "oci", "rootfs": "/fixture/rootfs"}})
    plan = {"runtime": str(runtime), "runtime_sha256": sha256(runtime / "source_state.json"),
            "tasks": [{"task_id": "task0", "task_file": str(task_path)}],
            "inputs_sha256": {str(task_path): sha256(task_path)}}
    task = Task.from_file(task_path)
    receipt = {"task_id": task.task_id, "task_sha256": sha256(task_path), "runtime": task.runtime.model_dump(),
               "python": {"version": "3.6.13", "executable": task.runtime.python},
               "caller_repository_unchanged": True, "probe_workspace_cleaned": True,
               "probe": {"returncode": 0, "timed_out": False, "filesystem_probe": {
                   "protocol": PROTOCOL, "checks": dict.fromkeys(REQUIRED_CHECKS, True),
                   "python_version": "3.6.13", "worker_sha256": sha256(worker), "cleanup_complete": True}}}
    return plan, task, receipt, sha256(worker)


@pytest.mark.parametrize("path,value,reason", [
    (("probe", "filesystem_probe", "protocol"), "old-protocol", "protocol"),
    (("probe", "filesystem_probe", "checks"), {}, "contract checks"),
    (("probe", "filesystem_probe", "checks", "v1.glob"), 1, "contract checks"),
    (("probe", "filesystem_probe", "worker_sha256"), "0" * 64, "worker code"),
    (("probe", "filesystem_probe", "python_version"), "3.12.0", "Python interpreter"),
    (("probe", "filesystem_probe", "cleanup_complete"), False, "cleanup"),
    (("task_id",), "another-task", "task identity"),
    (("task_sha256",), "0" * 64, "task identity"),
    (("runtime", "python"), "/different/bin/python", "different task runtime"),
    (("runtime", "rootfs"), "/different/rootfs", "different task runtime"),
    (("python", "executable"), "/usr/bin/python3", "worker interpreter"),
    (("python", "version"), "", "worker interpreter"),
    (("caller_repository_unchanged",), False, "preservation"),
    (("probe_workspace_cleaned",), False, "cleanup"),
    (("probe", "returncode"), 1, "sandbox probe"),
    (("probe", "timed_out"), True, "sandbox probe"),
])
def test_tool_receipt_requires_full_contract_and_expected_identity(tmp_path, path, value, reason):
    plan, task, receipt, worker_sha = tool_preflight_fixture(tmp_path)
    controller.validate_task_tool_receipt(receipt, task, receipt["task_sha256"], worker_sha)
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    task_sha = plan["inputs_sha256"][plan["tasks"][0]["task_file"]]
    with pytest.raises(ValueError, match=reason):
        controller.validate_task_tool_receipt(receipt, task, task_sha, worker_sha)


@pytest.mark.parametrize("action", ["run", "resume", "pipeline"])
def test_invalid_tool_receipt_aborts_before_any_gpu_or_model_start(tmp_path, monkeypatch, action):
    plan, task, receipt, worker_sha = tool_preflight_fixture(tmp_path)
    output, data = tmp_path / "output", tmp_path / "data"
    atomic_json(data / "preflight_report.json", {"ready_by_split": {"dev": True}})
    receipt["probe"]["filesystem_probe"]["checks"]["protected_write"] = False
    calls = []

    def launch(self, command, label, **kwargs):
        calls.append("tool_probe")
        assert command[command.index("--task") + 1] == plan["tasks"][0]["task_file"]
        atomic_json(Path(command[command.index("--receipt") + 1]), receipt)
        return {"label": label}

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU/model startup must wait for complete filesystem tool evidence")

    monkeypatch.setattr(controller.Controller, "launch", launch)
    monkeypatch.setattr(controller.Controller, "wait", lambda *a, **k: None)
    monkeypatch.setattr(controller, "plan_evaluation", lambda *a, **k: plan)
    monkeypatch.setattr(controller, "verify_resource_receipts", lambda: {})
    monkeypatch.setattr(controller, "resources", lambda *a: calls.append("resources"))
    monkeypatch.setattr(controller, "smoke", forbidden)
    monkeypatch.setattr(controller, "run_arm", forbidden)
    monkeypatch.setattr(controller, "start_service", forbidden)
    monkeypatch.setattr(controller, "wait_for_gpu_capacity", forbidden)
    monkeypatch.setattr(controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["run_scale_4gpu.py", action, "--output", str(output), "--data", str(data)])
    with pytest.raises(ValueError, match="Incomplete filesystem contract checks"):
        controller.main()
    assert calls == (["resources"] if action == "pipeline" else []) + ["tool_probe"]
    assert json.loads((output / "status.json").read_text())["stage"] == "failed"
    assert not list((output / "tool_preflight").glob("*/report.json"))


def test_tool_preflight_groups_identical_runtimes_and_reports_all_tasks(tmp_path):
    plan, task, receipt, worker_sha = tool_preflight_fixture(tmp_path)
    extra = tmp_path / "data/tasks/task1/task.json"
    atomic_json(extra, {**json.loads(Path(plan["tasks"][0]["task_file"]).read_text()), "task_id": "task1"})
    plan["tasks"].append({"task_id": "task1", "task_file": str(extra)})
    plan["inputs_sha256"][str(extra)] = sha256(extra)
    calls, updates = [], []

    def launch(command, label, **kwargs):
        calls.append(command)
        assert kwargs["env"]["PYTHONPATH"] == str(Path(plan["runtime"]) / "src")
        atomic_json(Path(command[command.index("--receipt") + 1]), receipt)
        return {}

    output = tmp_path / "output"
    c = SimpleNamespace(output=output, launch=launch, wait=lambda *a, **k: None,
                        update=lambda **fields: updates.append(fields))
    controller.preflight_tools(c, SimpleNamespace(), plan)
    assert len(calls) == 1
    report = json.loads(next((output / "tool_preflight").glob("*/report.json")).read_text())
    assert report["complete"] is True and report["task_count"] == 2
    assert report["receipts"][0]["task_ids"] == ["task0", "task1"]
    assert updates[-1]["stage"] == "tools_verified"
    assert updates[-1]["tool_preflight_covered_tasks"] == 2


@pytest.mark.parametrize("change", ["task_identity", "task_hash", "frozen_worker", "runtime_receipt"])
def test_tool_preflight_rejects_changed_inputs_before_launch(tmp_path, change):
    plan, task, receipt, worker_sha = tool_preflight_fixture(tmp_path)
    if change == "task_identity":
        plan["tasks"][0]["task_id"] = "other-task"
    elif change == "task_hash":
        Path(plan["tasks"][0]["task_file"]).write_text(json.dumps({"task_id": "task0", "repo": "repo", "problem": "changed"}))
    elif change == "frozen_worker":
        (Path(plan["runtime"]) / "src/screenshot_agent/worker.py").write_text("changed worker\n")
    else:
        plan["runtime_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed prepared/frozen inputs must fail before launching any child")

    c = SimpleNamespace(output=tmp_path / "output", launch=forbidden, update=forbidden)
    with pytest.raises(ValueError):
        controller.preflight_tools(c, SimpleNamespace(), plan)


@pytest.mark.sandbox
def test_probe_one_records_actual_interpreter_and_cleans_workspace(tmp_path):
    task_path, receipt_path = tmp_path / "task.json", tmp_path / "receipt.json"
    repo = tmp_path / "original"
    repo.mkdir()
    (repo / "original.py").write_text("original = 1\n")
    atomic_json(task_path, {"task_id": "fixture", "repo": str(repo), "problem": "fix"})
    task_sha = sha256(task_path)
    controller.probe_task_tools(task_path, receipt_path)
    receipt = json.loads(receipt_path.read_text())
    controller.validate_task_tool_receipt(receipt, Task.from_file(task_path), task_sha,
                                          sha256(ROOT / "src/screenshot_agent/worker.py"))
    assert receipt["python"]["executable"] == "/usr/bin/python3"
    assert receipt["caller_repository_unchanged"] is True
    assert receipt["probe_workspace_cleaned"] is True
    assert (repo / "original.py").read_text() == "original = 1\n"
    assert list(repo.iterdir()) == [repo / "original.py"]
    with pytest.raises(ValueError, match="overwrite"):
        controller.probe_task_tools(task_path, receipt_path)


@pytest.mark.sandbox
def test_preflight_checks_real_bug_and_gold_with_external_grader(tmp_path, monkeypatch):
    import screenshot_agent.scale_data as data_module

    root, output = tmp_path / "project", tmp_path / "data"
    (root / "vendor").mkdir(parents=True)
    (root / "vendor/SWE-bench").symlink_to(ROOT / "vendor/SWE-bench", target_is_directory=True)
    image_repo = tmp_path / "image/testbed"
    image_repo.mkdir(parents=True)
    (image_repo / "value.py").write_text("answer = 0\n")
    git(image_repo, "init", "-q")
    git(image_repo, "add", ".")
    git(image_repo, "commit", "-qm", "bug")
    base = git(image_repo, "rev-parse", "HEAD").stdout.decode().strip()
    patch = "diff --git a/value.py b/value.py\n--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-answer = 0\n+answer = 1\n"
    code = ("from value import answer\nprint(('PASSED' if answer == 1 else 'FAILED') + ' test_value.py::test_bug')\n"
            "print('PASSED test_value.py::test_keep')\nraise SystemExit(0 if answer == 1 else 1)\n")
    test_patch = ("diff --git a/test_value.py b/test_value.py\nnew file mode 100644\n--- /dev/null\n"
                  "+++ b/test_value.py\n@@ -0,0 +1,4 @@\n" + "".join("+" + s + "\n" for s in code.splitlines()))
    instance = "pytest-dev__pytest-fixture"
    upstream = root / "vendor/swe-bench-tasks/tasks" / instance
    upstream.mkdir(parents=True)
    (upstream / "test.patch").write_text(test_patch)
    (upstream / "gold.patch").write_text(patch)
    (upstream / "eval.sh").write_text(": '>>>>> Start Test Output'\npython3 test_value.py\n: '>>>>> End Test Output'\n")
    record = {"instance_id": instance, "repo": "pytest-dev/pytest", "sca_split": "dev", "base_commit": base,
              "image": "fixture/task:latest", "version": "4.4", "log_parser": "parse_log_pytest",
              "eval_type": "pass_and_fail", "problem_statement": "Return one.", "patch": patch, "test_patch": test_patch,
              "FAIL_TO_PASS": ["test_value.py::test_bug"], "PASS_TO_PASS": ["test_value.py::test_keep"]}
    atomic_json(output / "selected_records.json", [record])
    monkeypatch.setattr(data_module, "resolve_task_image", lambda *a, **k: ({
        "rootfs": str(image_repo.parent), "platform_digest": "sha256:" + "1" * 64,
        "requested_image": "fixture/task:latest"}, image_repo, {"environment_adapted": False}))
    # Exercise real isolated commands, real git patches and the pinned grader; no image download in a unit test.
    monkeypatch.setattr(data_module, "create_sandbox", lambda repo, events, protected, timeout, base, runtime:
                        BubblewrapSandbox(repo, events, protected, timeout, base))
    result = audit_one(root, output, instance, timeout=10)
    assert result["status"] == "validated", result
    assert not result["baseline"]["official"]["resolved"]
    assert result["reference"]["official"]["resolved"]
    assert (output / "tasks" / instance / "repo/value.py").read_text() == "answer = 0\n"
    assert result["repo_sha256"] == tree_hash(output / "tasks" / instance / "repo")
    assert result["reference_sha256"] == sha256(output / "tasks" / instance / "reference.patch")
