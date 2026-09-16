"""get_subtitles(time_ranges=[]) must fail, not silently OCR the whole file.

`ranges = time_ranges if time_ranges else [(time_start, time_end)]` treats
an explicitly-passed empty list the same as "no time_ranges argument at
all" (falsy), so a caller bug that produces `time_ranges=[]` would silently
OCR the entire video instead of failing. get_subtitles must instead raise
on an empty (but not-None) time_ranges, and do so before opening the video
at all -- these tests prove both.
"""
import pytest

from videocr import api


class _NeverReached(Exception):
    """Raised by a stubbed Video to prove the guard ran before any decode."""


def _stub_video(monkeypatch):
    def boom(*args, **kwargs):
        raise _NeverReached

    monkeypatch.setattr(api, "Video", boom)


def test_empty_time_ranges_raises_before_opening_the_video(monkeypatch):
    _stub_video(monkeypatch)

    with pytest.raises(ValueError, match="time_ranges"):
        api.get_subtitles("no-such-video.mkv", time_ranges=[])


def test_none_time_ranges_still_falls_back_to_time_start_time_end(monkeypatch):
    """The default (no time_ranges argument) is unaffected: it still reaches
    Video construction using time_start/time_end, exactly as before this
    validation was added."""
    _stub_video(monkeypatch)

    with pytest.raises(_NeverReached):
        api.get_subtitles("no-such-video.mkv", time_start="0:00", time_end="1:00")


def test_non_empty_time_ranges_is_unaffected(monkeypatch):
    _stub_video(monkeypatch)

    with pytest.raises(_NeverReached):
        api.get_subtitles("no-such-video.mkv", time_ranges=[("0:00", "1:00")])
