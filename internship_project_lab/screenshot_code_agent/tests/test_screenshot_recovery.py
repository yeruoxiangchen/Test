import json

import pytest

from screenshot_agent.config import Limits, ScreenshotPolicy
from screenshot_agent.control import Budget, Events
from screenshot_agent.images import ImageStore, Reading, parse_vision_response
from screenshot_agent.tools import build_tools


def test_literal_newlines_are_recovered_without_changing_source():
    source='def f():\n    return "中文"\n'
    raw=json.dumps({'raw_text':source,'kind':'code'},ensure_ascii=False).replace('\\n','\n')
    reading=parse_vision_response('```json\n'+raw+'\n```')
    assert reading.raw_text==source
    assert reading.format_status=='relaxed_json'
    assert reading.uncertainties


@pytest.mark.parametrize('text',['{"raw_text":"unfinished','{"raw_text":"quote"inside"}','plain code'])
def test_ambiguous_responses_are_kept_verbatim(text):
    result=parse_vision_response(text)
    assert result.raw_text==text and result.format_status=='unparsed_text'


def test_strict_json_does_not_decode_code_escapes_twice():
    source=r'pattern = "\\n"'
    result=parse_vision_response(json.dumps({'raw_text':source}))
    assert result.raw_text==source and result.format_status=='strict_json'


def test_crop_policy_preserves_repository_tools(sandbox,tmp_path):
    events=Events(tmp_path/'events.jsonl')
    budget=Budget(Limits(),events)
    store=ImageStore([],tmp_path/'shots',events)
    class Reader:
        def read(self,image_id):
            return Reading(raw_text='test')
    args=(sandbox,store,Reader(),budget,events,'true')
    enabled=build_tools(*args)
    disabled=build_tools(*args,screenshot_policy=ScreenshotPolicy(allow_crop=False))
    assert {t.name for t in enabled}-{t.name for t in disabled}=={'crop_screenshot'}
