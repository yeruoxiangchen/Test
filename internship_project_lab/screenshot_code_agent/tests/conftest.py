import json

import pytest

from screenshot_agent.control import Events
from screenshot_agent.sandbox import BubblewrapSandbox


@pytest.fixture
def sandbox(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("value = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_fixed.py").write_text("immutable\n")
    sandbox = BubblewrapSandbox(repo, Events(tmp_path / "events.jsonl"), ["tests"], timeout=5)
    sandbox.probe()  # fail loudly if isolation is unavailable; never silently skip it
    yield sandbox
    sandbox.cancel()


def tool_result(value):
    return json.loads(json.loads(value)["output"])
