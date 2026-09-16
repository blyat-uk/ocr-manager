import sys

import tools.fidelity_check as fc
from tools.fidelity_check import Case, digest, load_cases, verify_case

ASS_A = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,你好
"""
ASS_B = ASS_A.replace("你好", "你們")
ASS_C = ASS_A.replace("Title: x", "Title: y")

# A case name that actually exists in tests/fixtures/fidelity_cases.json, so
# these tests exercise load_cases()/media_root() for real (both are cheap
# JSON reads — no media, no GPU) while run_case() itself is monkeypatched.
REAL_CASE_NAME = "slay_1080p_dialogue"


def test_digest_detects_text_change():
    assert digest(ASS_A) != digest(ASS_B)


def test_digest_ignores_header_only_change():
    assert digest(ASS_A) == digest(ASS_C)


def test_cases_load_and_are_named_uniquely():
    cases = load_cases()
    assert cases and all(isinstance(c, Case) for c in cases)
    names = [c.name for c in cases]
    assert len(names) == len(set(names))


def _dummy_case(name="c") -> Case:
    return Case(
        name=name, project="p", file="f.mp4",
        time_ranges=[["0:00", "0:01"]], crop=[0, 0, 1, 1],
        brightness=1, detect_labels=False,
    )


def _ass(text: str) -> str:
    return (
        "[Events]\n"
        f"Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{text}\n"
    )


# --- verify_case(): the pure compare/report function, no media, no GPU ---

def test_verify_case_ok_writes_nothing(tmp_path):
    case = _dummy_case()
    ass = _ass("hi")
    (tmp_path / "c.ass").write_text(ass, encoding="utf-8")

    verdict, message = verify_case(case, ass, tmp_path)

    assert verdict == "OK"
    assert "OK   c" in message
    assert not (tmp_path / "c.actual.ass").exists()


def test_verify_case_fail_writes_actual_with_produced_content(tmp_path):
    case = _dummy_case()
    (tmp_path / "c.ass").write_text(_ass("hi"), encoding="utf-8")
    produced = _ass("bye")

    verdict, message = verify_case(case, produced, tmp_path)

    assert verdict == "FAIL"
    assert "FAIL c" in message
    actual = tmp_path / "c.actual.ass"
    assert actual.exists()
    assert actual.read_text(encoding="utf-8") == produced


# --- main(): the CLI loop, exercised end-to-end with run_case mocked out ---

def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fidelity_check.py", *args])


def test_main_missing_golden_is_a_failure_and_never_calls_run_case(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fc, "GOLDEN_DIR", tmp_path)  # empty: no golden written

    def _must_not_run(case, root):
        raise AssertionError("run_case must not be called before the golden-existence check")
    monkeypatch.setattr(fc, "run_case", _must_not_run)
    _set_argv(monkeypatch, "--case", REAL_CASE_NAME)

    rc = fc.main()

    assert rc != 0
    assert f"MISSING GOLDEN {REAL_CASE_NAME}" in capsys.readouterr().out


def test_main_skip_on_missing_input_is_not_a_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fc, "GOLDEN_DIR", tmp_path)
    (tmp_path / f"{REAL_CASE_NAME}.ass").write_text(_ass("hi"), encoding="utf-8")

    def _no_video(case, root):
        raise FileNotFoundError(root / case.project / case.file)
    monkeypatch.setattr(fc, "run_case", _no_video)
    _set_argv(monkeypatch, "--case", REAL_CASE_NAME)

    rc = fc.main()

    assert rc == 0
    assert f"SKIP {REAL_CASE_NAME}" in capsys.readouterr().out


def test_main_ok_then_fail_through_the_full_loop(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fc, "GOLDEN_DIR", tmp_path)
    golden_ass = _ass("hi")
    (tmp_path / f"{REAL_CASE_NAME}.ass").write_text(golden_ass, encoding="utf-8")
    _set_argv(monkeypatch, "--case", REAL_CASE_NAME)

    monkeypatch.setattr(fc, "run_case", lambda case, root: golden_ass)
    rc = fc.main()
    assert rc == 0
    assert f"OK   {REAL_CASE_NAME}" in capsys.readouterr().out
    assert not (tmp_path / f"{REAL_CASE_NAME}.actual.ass").exists()

    mismatched = _ass("bye")
    monkeypatch.setattr(fc, "run_case", lambda case, root: mismatched)
    rc = fc.main()
    assert rc != 0
    assert f"FAIL {REAL_CASE_NAME}" in capsys.readouterr().out
    actual = tmp_path / f"{REAL_CASE_NAME}.actual.ass"
    assert actual.exists()
    assert actual.read_text(encoding="utf-8") == mismatched
