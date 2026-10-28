import json
from pathlib import Path

import pytest

from screenshot_agent.benchmark import official_grade

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, f2p, p2p, parser="parse_log_pydicom"):
    checkout = ROOT / "vendor/SWE-bench"
    if not (checkout / "swebench/harness/grading.py").is_file():
        pytest.skip("Pinned official harness is required")
    metadata = tmp_path / "benchmark.json"
    metadata.write_text(json.dumps({"harness_checkout": str(checkout), "record": {
        "instance_id": "canonical-id-fixture", "image": "unused", "repo": "pydicom/pydicom", "version": "1.3",
        "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p, "log_parser": parser, "eval_type": "pass_and_fail"}}))
    return metadata


def write_test_log(tmp_path, *lines):
    log = tmp_path / "test.log"
    log.write_text(">>>>> Start Test Output\n" + "\n".join(lines)
                   + "\n>>>>> End Test Output\n>>>>> Test Exit Code: 0\n")
    return log


@pytest.mark.parametrize("parser", ["parse_log_pytest_options", "parse_log_pydicom", "parse_log_requests"])
def test_expected_paths_use_the_same_official_parser_and_retain_raw_ids(tmp_path, parser):
    bug = "test_pixels.py::test_bug[/old/worker/data/a.dcm]"
    keep = "test_pixels.py::test_keep[/old/worker/data/b.dcm]"
    metadata = fixture(tmp_path, json.dumps([bug]), json.dumps([keep]), parser)
    original = metadata.read_bytes()
    result = official_grade(metadata, write_test_log(tmp_path,
        "PASSED test_pixels.py::test_bug[/testbed/data/a.dcm]",
        "PASSED test_pixels.py::test_keep[/testbed/data/b.dcm]"))
    assert result["resolved"]
    assert result["expected_fail_to_pass"] == ["test_pixels.py::test_bug[/a.dcm]"]
    normalization = result["expected_id_normalization"]
    assert normalization["raw"] == {"FAIL_TO_PASS": [bug], "PASS_TO_PASS": [keep]}
    assert normalization["counts_preserved"] and len(normalization["changes"]) == 2
    assert metadata.read_bytes() == original
    assert result["grading_source"].startswith(str(ROOT / "vendor/SWE-bench"))


@pytest.mark.parametrize("bug_status,keep_status", [("FAILED", "PASSED"), ("SKIPPED", "PASSED"),
                                                    (None, "PASSED"), ("PASSED", "FAILED")])
def test_normalization_never_substitutes_for_actual_passing_tests(tmp_path, bug_status, keep_status):
    metadata = fixture(tmp_path, ["t.py::bug[/old/a]"], ["t.py::keep[/old/b]"])
    lines = [f"{keep_status} t.py::keep[/new/b]"]
    if bug_status:
        lines.append(f"{bug_status} t.py::bug[/new/a]")
    result = official_grade(metadata, write_test_log(tmp_path, *lines))
    assert not result["resolved"]


@pytest.mark.parametrize("cross_group", [False, True])
def test_colliding_expected_ids_are_rejected_not_merged(tmp_path, cross_group):
    first, second = "t.py::bug[/one/shared.dcm]", "t.py::bug[/two/shared.dcm]"
    metadata = fixture(tmp_path, [first] if cross_group else [first, second], [second] if cross_group else [])
    with pytest.raises(RuntimeError, match="Expected test ID collision"):
        official_grade(metadata, write_test_log(tmp_path, "PASSED t.py::bug[/shared.dcm]"))


def test_other_parsers_do_not_receive_pytest_options_normalization(tmp_path):
    metadata = fixture(tmp_path, ["t.py::bug[/old/a]"], [], parser="parse_log_pytest")
    result = official_grade(metadata, write_test_log(tmp_path, "PASSED t.py::bug[/new/a]"))
    assert not result["resolved"]
    assert result["expected_id_normalization"] is None
    assert result["expected_fail_to_pass"] == ["t.py::bug[/old/a]"]


def test_official_exceptions_for_unc_and_wildcard_parameters_are_preserved(tmp_path):
    names = ["t.py::unc[//server/dir/a]", "t.py::wildcard[/a/*/b]", "t.py::plain[value]"]
    result = official_grade(fixture(tmp_path, names, []), write_test_log(tmp_path, *("PASSED " + n for n in names)))
    assert result["resolved"]
    assert result["expected_fail_to_pass"] == names
    assert result["expected_id_normalization"]["changes"] == []
