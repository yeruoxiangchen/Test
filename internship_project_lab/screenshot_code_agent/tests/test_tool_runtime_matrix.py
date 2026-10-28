import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "tool_runtime_matrix", Path(__file__).resolve().parents[1] / "scripts/check_tool_runtime_compat.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def receipt():
    return {"protocol": "filesystem_tools_v1", "python_version": "3.8.20",
            "worker_sha256": "fixed-worker", "cleanup_complete": True,
            "checks": {name: True for name in matrix.REQUIRED_CHECKS}}


def test_full_contract_receipt_is_required_not_only_a_python_import():
    matrix.check_receipt(receipt(), "3.8.20", "fixed-worker")
    with pytest.raises(ValueError, match="protocol receipt"):
        matrix.check_receipt({"returncode": 0, "output": "sandbox-ok /testbed"}, "3.8.20", "fixed-worker")


@pytest.mark.parametrize("missing", ["protected_write", "v2.edit_file", "symlink_escape"])
def test_missing_contract_checks_cannot_report_a_green_matrix(missing):
    value = receipt()
    del value["checks"][missing]
    with pytest.raises(ValueError, match="Incomplete filesystem contract"):
        matrix.check_receipt(value, "3.8.20", "fixed-worker")


def test_ast_feedback_must_come_from_target_interpreter_and_current_worker():
    with pytest.raises(ValueError, match="actual task Python"):
        matrix.check_receipt(receipt(), "3.12.0", "fixed-worker")
    with pytest.raises(ValueError, match="different worker"):
        matrix.check_receipt(receipt(), "3.8.20", "old-worker")


def test_disposable_probe_cleanup_must_be_confirmed():
    value = receipt()
    value["cleanup_complete"] = False
    with pytest.raises(ValueError, match="cleanup"):
        matrix.check_receipt(value, "3.8.20", "fixed-worker")
