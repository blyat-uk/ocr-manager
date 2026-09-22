"""Cross-platform hardening for the packaged app (Linux, Windows, macOS).

Windows is simulated on Linux: sys.platform is patched for the pieces that
branch on it, and the `windows_text_mode` fixture (tests/conftest.py) gives
open() Windows' newline translation.

- Every child process in app/, core/ and videocr/ is started with
  core.proc.hidden_child() (no console window flashing up under pythonw.exe),
  and every text-mode one decodes UTF-8 whatever the locale.
- ASS output is LF on every OS, so the bytes match the goldens everywhere.
- The QA pass still loads its dictionary when fd 2 is not usable.
- Video discovery ignores the extension's case.
- The run-end notification is per-OS and never waited for on the GUI thread.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from app import notify
from core import ass_qafix
from core.jobs import detect_jobs
from core.proc import TEXT_ENCODING, hidden_child
from core.project import store
from videocr import api

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
ASS = "[Script Info]\nScriptType: v4.00+\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" \
      "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,字幕\n"


# --------------------------------------------------------------------------
# Child processes
# --------------------------------------------------------------------------

def test_hidden_child_is_create_no_window_on_windows_and_nothing_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert hidden_child() == {}
    monkeypatch.setattr(sys, "platform", "darwin")
    assert hidden_child() == {}
    monkeypatch.setattr(sys, "platform", "win32")
    assert hidden_child() == {"creationflags": 0x08000000}          # subprocess.CREATE_NO_WINDOW
    assert TEXT_ENCODING == {"encoding": "utf-8", "errors": "replace"}


def _subprocess_calls():
    for root in ("app", "core", "videocr"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and (node.func.value.id, node.func.attr) in
                        {("subprocess", name) for name in SUBPROCESS_CALLS} | {("os", "system"), ("os", "popen")}):
                    yield f"{path.relative_to(REPO_ROOT)}:{node.lineno}", node


def _spreads(node: ast.Call, name: str) -> bool:
    """`**name()` or `**name` among the call's keywords."""
    for keyword in node.keywords:
        if keyword.arg is None:
            value = keyword.value.func if isinstance(keyword.value, ast.Call) else keyword.value
            if isinstance(value, ast.Name) and value.id == name:
                return True
    return False


def test_every_child_process_is_started_hidden_and_text_is_read_as_utf8():
    calls = list(_subprocess_calls())
    assert len(calls) >= 9                  # the scan sees the ffmpeg/ffprobe/notifier call sites
    offenders = []
    for where, node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.value.id == "os":
            offenders.append(f"{where}: os.{node.func.attr} (use subprocess with hidden_child())")
            continue
        if not _spreads(node, "hidden_child"):
            offenders.append(f"{where}: no **hidden_child()")
        text = any(keyword.arg in ("text", "universal_newlines") for keyword in node.keywords)
        encoded = _spreads(node, "TEXT_ENCODING") or any(keyword.arg == "encoding" for keyword in node.keywords)
        if text and not encoded:
            offenders.append(f"{where}: text mode without encoding=\"utf-8\"")
    assert offenders == []


# --------------------------------------------------------------------------
# LF output on every OS
# --------------------------------------------------------------------------

def test_the_windows_text_mode_fixture_really_writes_crlf(tmp_path, windows_text_mode):
    """Guards the tests below: without newline="\\n" the fixture does make CRLF."""
    path = tmp_path / "plain.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("a\nb\n")
    assert path.read_bytes() == b"a\r\nb\r\n"


def test_save_subtitles_to_file_writes_lf(tmp_path, monkeypatch, windows_text_mode):
    monkeypatch.setattr(api, "get_subtitles", lambda *args, **kwargs: ASS)
    path = tmp_path / "out.ass"
    api.save_subtitles_to_file("video.mkv", file_path=str(path))
    assert path.read_bytes() == ASS.encode("utf-8")


def test_the_qa_pass_writes_lf(tmp_path, windows_text_mode):
    path = tmp_path / "out.ass"
    path.write_bytes(ASS.encode("utf-8"))
    ass_qafix.process_file(str(path))
    data = path.read_bytes()
    assert b"\r\n" not in data and "字幕".encode("utf-8") in data


def test_the_proof_qa_pass_reads_back_what_a_run_would_write(windows_text_mode):
    assert "\r" not in detect_jobs.qa_pass(ASS)


# --------------------------------------------------------------------------
# The QA pass's dictionary without a usable fd 2
# --------------------------------------------------------------------------

@pytest.fixture
def fake_enchant(monkeypatch):
    module = types.ModuleType("enchant")
    module.Dict = lambda tag: ("dictionary", tag)
    monkeypatch.setitem(sys.modules, "enchant", module)
    monkeypatch.setattr(ass_qafix, "_enchant_dict", None)
    return module


def test_the_dictionary_loads_when_fd_2_cannot_be_duplicated(fake_enchant, monkeypatch):
    def no_fd(fd):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(os, "dup", no_fd)
    assert ass_qafix._get_enchant_dict() == ("dictionary", "en_US")


def test_the_dictionary_loads_with_fd_2_silenced_and_restored(fake_enchant):
    before = os.fstat(2)
    assert ass_qafix._get_enchant_dict() == ("dictionary", "en_US")
    assert os.fstat(2) == before


# --------------------------------------------------------------------------
# Video discovery
# --------------------------------------------------------------------------

def test_video_discovery_ignores_the_extension_case_and_keeps_its_order(tmp_path):
    for name in ("ep02.mkv", "EP01.MKV", "ep03.Mp4", "a.b.mp4", "notes.txt", "ep04.mkv.part", "mkv"):
        (tmp_path / name).write_bytes(b"")
    assert store.list_video_files(str(tmp_path)) == sorted(["EP01.MKV", "a.b.mp4", "ep02.mkv", "ep03.Mp4"])


# --------------------------------------------------------------------------
# The run-end notification
# --------------------------------------------------------------------------

def test_the_notifier_command_per_os():
    assert notify.notify_command("OCR Complete", "Finished", "normal", "linux") == \
        ["notify-send", "-a", "OCR Manager", "-u", "normal", "OCR Complete", "Finished"]
    assert notify.notify_command("OCR Failed", 'a "quoted" \\ body', "critical", "darwin") == [
        "osascript", "-e",
        'display notification "a \\"quoted\\" \\\\ body" with title "OCR Manager" subtitle "OCR Failed"']
    assert notify.notify_command("OCR Complete", "Finished", "normal", "win32") is None


class _Child:
    """A notifier that runs until released (or times out every wait)."""

    def __init__(self, hang: bool = False):
        self.released = threading.Event()
        self.reaped = threading.Event()
        self.killed = False
        self.hang = hang

    def wait(self, timeout=None):
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("notifier", timeout)
        self.released.wait(5)
        self.reaped.set()
        return 0

    def kill(self):
        self.killed = True
        self.released.set()


def test_the_notifier_is_started_hidden_and_reaped_off_the_calling_thread(monkeypatch):
    child, started = _Child(), []

    def fake_popen(args, **kwargs):
        started.append((list(args), kwargs))
        return child

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    notify.send_notification("OCR Complete", "Finished", "normal")      # returns while the child runs
    assert not child.reaped.is_set()
    (args, kwargs), = started
    assert args[0] == "notify-send" and kwargs["stdout"] == subprocess.DEVNULL
    child.released.set()
    assert child.reaped.wait(5)


def test_a_hung_notifier_is_killed_and_a_missing_one_ignored(monkeypatch, caplog):
    child = _Child(hang=True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: child)
    notify.send_notification("OCR Complete", "Finished", "normal")
    assert child.reaped.wait(5) and child.killed

    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(subprocess, "Popen", missing)
    notify.send_notification("OCR Complete", "Finished", "normal")


def test_windows_starts_no_notifier(monkeypatch):
    def refuse(args, **kwargs):
        raise AssertionError("no notifier is started on Windows")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "Popen", refuse)
    notify.send_notification("OCR Complete", "Finished", "normal")
