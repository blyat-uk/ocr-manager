"""The Folder settings language list is every language PaddleOCR can read.

`LANGUAGE_NAMES` is static (a frozen build installs Paddle on first run, so
the sheet cannot ask it). This test reads the installed paddleocr's own
language lists out of `_get_ocr_model_names` and fails when the two drift
apart, e.g. after a paddleocr upgrade adds or drops a language. It also
checks every listed language names its output with a well-formed tag
(core.project.languages.output_tag).
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest

from app.views.folder_settings_fields import LANGUAGE_NAMES, LANGUAGES, language_label
from core.project.languages import output_tag

TAG = re.compile(r"^[a-z]{2,3}(-([A-Z][a-z]{3}|[A-Z]{2}))?$")      # language, then a script or a region

ALIASES = {"french": "fr", "german": "de"}      # Paddle accepts both spellings; the list shows one


def paddle_languages() -> set[str]:
    """Every `lang` string named in `_get_ocr_model_names` that resolves to a recognition model."""
    ocr = pytest.importorskip("paddleocr._pipelines.ocr")      # only this needs paddle installed
    source = textwrap.dedent(inspect.getsource(ocr.PaddleOCR._get_ocr_model_names))
    named = {node.value for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    return {lang for lang in named if ocr.PaddleOCR._get_ocr_model_names(None, lang, None)[1] is not None}


def test_the_list_is_exactly_the_languages_paddle_reads():
    assert set(LANGUAGE_NAMES) == paddle_languages() - set(ALIASES)
    assert set(ALIASES.values()) <= set(LANGUAGE_NAMES)


def test_the_list_is_sorted_by_name_and_every_label_is_unique():
    labels = [language_label(code) for code in LANGUAGES]
    assert sorted(LANGUAGES) == sorted(LANGUAGE_NAMES)
    assert labels == sorted(labels, key=str.casefold)
    assert len(set(labels)) == len(labels)
    assert len({name.casefold() for name in LANGUAGE_NAMES.values()}) == len(LANGUAGE_NAMES)


@pytest.mark.parametrize("code", sorted(LANGUAGE_NAMES))
def test_each_language_names_its_output_with_a_well_formed_tag(code):
    assert TAG.match(output_tag(code)), (code, output_tag(code))


@pytest.mark.parametrize(("code", "tag"), [
    ("ch", "zh"), ("chinese_cht", "zh-Hant"), ("japan", "ja"), ("korean", "ko"),
    ("en", "en"), ("rs_latin", "sr-Latn"),
])
def test_output_tag_literals(code, tag):
    assert output_tag(code) == tag


@pytest.mark.parametrize(("typed", "tag"), [
    ("../x", "x"), ("a/b", "a_b"), ("a\\b", "a_b"), ("Chinese (Simplified)", "Chinese_Simplified"),
    ("..", "zh"), ("/", "zh"), ("", "zh"), ("   ", "zh"), (None, "zh"),
])
def test_a_stored_language_that_is_no_plain_code_still_gives_one_safe_path_component(typed, tag):
    assert output_tag(typed) == tag
