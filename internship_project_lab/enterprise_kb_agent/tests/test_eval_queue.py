import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def queue_module():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_queue.py"
    spec = importlib.util.spec_from_file_location("eval_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fail", [False, True])
def test_report_only_never_runs_inference_and_records_outcome(tmp_path, monkeypatch, capsys, fail):
    queue = queue_module()
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        assert args[args.index("kb.benchmark") + 1] == "compare"
        if fail:
            raise subprocess.CalledProcessError(1, args)
        folder = tmp_path / "comparison"
        folder.mkdir()
        (folder / "comparison.json").write_text("{}")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(queue.subprocess, "run", command)
    monkeypatch.setattr(sys, "argv", ["eval_queue.py", "report", "--output", str(tmp_path)])
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            queue.main()
    else:
        queue.main()
    assert len(calls) == 1
    state = json.loads((tmp_path / "queue_state.json").read_text())
    assert state["status"] == ("failed" if fail else "complete")
    monkeypatch.setattr(sys, "argv", ["eval_queue.py", "status", "--output", str(tmp_path)])
    queue.main()
    output = capsys.readouterr().out
    assert "总状态: " + state["status"] in output
    if fail:
        assert "pending" not in output.split("配对报告:")[1]


def test_old_exited_queue_does_not_report_pending_forever(tmp_path, monkeypatch, capsys):
    queue = queue_module()
    monkeypatch.setattr(sys, "argv", ["eval_queue.py", "status", "--output", str(tmp_path)])
    queue.main()
    assert "未完成；队列已退出" in capsys.readouterr().out
