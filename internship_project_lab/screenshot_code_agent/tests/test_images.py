import json

import pytest
from PIL import Image

from screenshot_agent.config import ImageInput
from screenshot_agent.control import Events, public_payload
from screenshot_agent.data import render_screenshot
from screenshot_agent.images import ImageStore, OracleReader, VisionReader


@pytest.fixture
def store(tmp_path):
    image = tmp_path / "source.png"
    render_screenshot("miniapp.py\n\ndef add(a, b):\n    return a - b\n", image)
    return ImageStore([ImageInput(image_id="shot", path=str(image), reference=str(image.with_suffix(".txt")),
                      layout=str(image.with_suffix(".layout.json")))], tmp_path / "inputs", Events(tmp_path / "events.jsonl"))


def test_registry_does_not_expose_transcription(store):
    public = json.dumps(store.public())
    assert "reference" not in public
    assert "return a - b" not in public
    assert "path" not in public
    with pytest.raises(ValueError):
        store.get("../../source.png")


def test_crop_is_deterministic_and_stays_inside_image(store):
    entry = store.get("shot")
    bbox = [0, 55, entry["width"], entry["height"]]
    crop = store.crop("shot", bbox)
    assert crop == store.crop("shot", bbox)
    with Image.open(store.get(crop["image_id"])["path"]) as image:
        assert image.size == (entry["width"], entry["height"]-55)
    text = OracleReader(store).read(crop["image_id"]).raw_text
    assert "miniapp.py" not in text
    assert "return a - b" in text
    for invalid in [[0, 0, 0, 5], [-1, 0, 5, 5], [0, 0, entry["width"]+1, 5], [False, 0, 5, 5]]:
        with pytest.raises(ValueError):
            store.crop("shot", invalid)


def test_oracle_is_visible_text_only(store):
    value = OracleReader(store).read("shot")
    assert value.file_hints == ["miniapp.py"]
    assert "return a - b" in value.raw_text
    assert "return a + b" not in value.raw_text


def test_vision_receives_image_without_reference(store):
    class Fake:
        def complete(self, messages):
            text = json.dumps(messages)
            assert "data:image/png;base64," in text
            assert "return a - b" not in text
            assert "reference" not in text
            class Message:
                content = '{"raw_text":"observed", "uncertainties":["ambiguous symbol"]}'
            class Choice:
                message = Message()
            class Response:
                choices = [Choice()]
            return Response()
    reading = VisionReader(store, Fake()).read("shot")
    assert reading.raw_text == "observed"
    assert reading.uncertainties == ["ambiguous symbol"]


def test_trace_does_not_repeat_base64():
    value = public_payload({"content": [{"image_url": {"url": "data:image/png;base64,VERY-LONG"}}]})
    assert "VERY-LONG" not in json.dumps(value)
    assert "image_payload_sha256" in json.dumps(value)
