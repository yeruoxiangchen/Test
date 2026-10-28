import json

import pytest

from screenshot_agent.control import Events
from screenshot_agent.report import make_report
from screenshot_agent.repository import export_patch, prepare_repository, replay_patch


def test_grep_accepts_a_literal_file_path(sandbox):
    data=json.loads(sandbox.filesystem("grep",{"pattern":"value", "path":"sample.py"}))
    assert data["returncode"]==0
    assert "sample.py:1:value" in json.loads(data["output"])["output"]


def test_grep_rejects_missing_or_glob_paths_with_actionable_feedback(sandbox):
    data=json.loads(sandbox.filesystem("grep",{"pattern":"value", "path":"*.py"}))
    assert data["returncode"]!=0
    assert "literal file/directory" in json.loads(data["output"])["error"]


def test_patch_omits_generated_caches_and_preserves_new_deleted_source(tmp_path):
    source=tmp_path/"input"
    source.mkdir()
    (source/"app.py").write_text("value = 1\n")
    (source/"old.py").write_text("old = True\n")
    base,workspace=prepare_repository(source,tmp_path/"run")
    (workspace/"app.py").write_text("value = 2\n")
    (workspace/"old.py").unlink()
    (workspace/"new.py").write_text("new = True\n")
    for directory in ['.pytest_cache','pkg/__pycache__']:
        p=workspace/directory
        p.mkdir(parents=True)
        (p/"generated").write_text("cache data")
    patch=tmp_path/"patch.diff"
    result=export_patch(base,workspace,patch,[])
    assert set(result['changed_files'])=={'app.py','old.py','new.py'}
    assert b'__pycache__' not in patch.read_bytes() and b'.pytest_cache' not in patch.read_bytes()
    assert replay_patch(base,tmp_path/"verification",patch)['applied']
    assert (tmp_path/"verification/app.py").read_text()=="value = 2\n"
    assert not (tmp_path/"verification/old.py").exists()


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_report_preserves_unicode_line_separators_inside_events(tmp_path, separator):
    reference = tmp_path / "reference.txt"
    text = "before" + separator + "after"
    reference.write_text(text)
    (tmp_path/"task.json").write_text(json.dumps({"images":[{"image_id":"shot", "reference":str(reference)}]}))
    row = {"task_id":"unicode", "group_id":"repo", "split":"dev", "provenance":"test",
           "reader":"vlm", "code_backend":"api", "resolved":False, "status":"unresolved",
           "usage":{"vision_model_calls":1}}
    (tmp_path/"result.json").write_text(json.dumps(row))
    events = Events(tmp_path/"events.jsonl")
    events.emit("sandbox_command", output=text)
    events.emit("tool_start", name="read_screenshot")
    events.emit("model_request", kind="vision")
    events.emit("screenshot_read", image_id="shot", reading={"raw_text":text})
    events.emit("run_finished", result=row)
    report = make_report(tmp_path)
    assert report["records"][0]["reading_metrics"]["readings"][0]["exact"]
    assert report["records"][0]["usage"]["screenshot_read_calls"] == 1
    assert report["finished_runs"] == 1
