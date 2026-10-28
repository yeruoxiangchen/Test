import importlib.util
from pathlib import Path

import pytest

from screenshot_agent.data import render_screenshot

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/audit_screenshot_glyphs.py"
SPEC = importlib.util.spec_from_file_location("glyph_audit",SCRIPT)
glyph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(glyph)


def test_font_audit_detects_visible_math_and_cjk_gaps():
    mono = glyph.charset(glyph.MONO)
    assert glyph.missing_glyphs("print('ℯ')",mono)==["ℯ"]
    assert glyph.missing_glyphs("中文",mono)
    assert not glyph.missing_glyphs("ℯ",glyph.charset(glyph.SYMBOL))
    assert not glyph.missing_glyphs("\ufe0f \n",mono)


def test_font_fix_refuses_to_mutate_already_exposed_case(tmp_path):
    dataset, runs = tmp_path/"dataset", tmp_path/"runs"
    folder = dataset/"case"
    folder.mkdir(parents=True)
    (folder/"screenshot_light.txt").write_text("print('ℯ')\n")
    (runs/"case_oracle").mkdir(parents=True)
    with pytest.raises(ValueError,match="already exposed"):
        glyph.repair_audit(dataset,runs,tmp_path/"backup")
    assert not (tmp_path/"backup").exists()


def test_font_fix_preserves_text_and_becomes_idempotent_after_exposure(tmp_path):
    dataset, runs = tmp_path/"dataset", tmp_path/"runs"
    folder = dataset/"case"
    folder.mkdir(parents=True)
    for theme in ["light","dark"]:
        render_screenshot("print('ℯ')\n",folder/f"screenshot_{theme}.png",theme)
    reference = (folder/"screenshot_light.txt").read_bytes()
    before = (folder/"screenshot_light.png").read_bytes()
    report = glyph.repair_audit(dataset,runs,tmp_path/"backup")
    assert report["all_visible_glyphs_supported"]
    assert report["tasks"]["case"]["correction_before_model_exposure"]
    assert (folder/"screenshot_light.txt").read_bytes()==reference
    assert (folder/"screenshot_light.png").read_bytes()!=before
    after = (folder/"screenshot_light.png").read_bytes()
    (runs/"case_oracle").mkdir(parents=True)
    glyph.repair_audit(dataset,runs,tmp_path/"backup")
    assert (folder/"screenshot_light.png").read_bytes()==after
