from tools.fidelity_check import Case, digest, load_cases

ASS_A = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,你好
"""
ASS_B = ASS_A.replace("你好", "你們")
ASS_C = ASS_A.replace("Title: x", "Title: y")


def test_digest_detects_text_change():
    assert digest(ASS_A) != digest(ASS_B)


def test_digest_ignores_header_only_change():
    assert digest(ASS_A) == digest(ASS_C)


def test_cases_load_and_are_named_uniquely():
    cases = load_cases()
    assert cases and all(isinstance(c, Case) for c in cases)
    names = [c.name for c in cases]
    assert len(names) == len(set(names))
