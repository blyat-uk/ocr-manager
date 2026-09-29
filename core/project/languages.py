"""The language tag an output subtitle file is named with.

`FolderSettings.ocr_lang` is PaddleOCR's own language value ("ch", "japan",
"rs_latin", ...). Outputs are named with the standard tag for it instead
(`<tag>/<stem>.<tag>.ass`), which players read as the subtitle's language:
"ch" writes `zh/EP01.zh.ass`.

Most Paddle values already are ISO 639 codes; TAGS lists only the ones that
are not, or that name a script. A value missing from it is its own tag.
`tests/test_language_names.py` checks every language the settings list
offers comes out as a well-formed tag.

ocr_lang is also whatever the user typed into the settings sheet, and the tag
becomes a directory and part of a file name, so anything but letters,
digits, "-" and "_" is reduced away: the tag is always one plain path
component ("../x" gives "x", never a path out of the folder).
"""
from __future__ import annotations

import re

DEFAULT_TAG = "zh"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")

TAGS = {
    "ch": "zh",
    "chinese_cht": "zh-Hant",
    "japan": "ja",
    "korean": "ko",
    "french": "fr",               # Paddle's aliases of fr and de
    "german": "de",
    "rs_latin": "sr-Latn",
    "rs_cyrillic": "sr-Cyrl",
    "ava": "av",                  # Avar
    "che": "ce",                  # Chechen
    "mo": "ro-MD",                # Moldovan: "mo" is a retired code
    "ang": "anp",                 # Paddle's "ang" is Angika (its Devanagari model), not Old English
    "mah": "mag",                 # Paddle's "mah" is Magahi, not Marshallese
}


def output_tag(ocr_lang: str) -> str:
    """The tag outputs are named with for the OCR language `ocr_lang`: one
    safe path component, DEFAULT_TAG when nothing usable is left."""
    code = str(ocr_lang or "").strip()
    return _UNSAFE.sub("_", TAGS.get(code, code)).strip("_-") or DEFAULT_TAG
