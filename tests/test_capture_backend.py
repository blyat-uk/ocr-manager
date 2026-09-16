import pytest

from videocr import pyav_adapter


def test_pyav_is_available():
    assert pyav_adapter.PYAV_AVAILABLE, (
        f"PyAV failed to import: {pyav_adapter.PYAV_IMPORT_ERROR}"
    )


def test_backend_is_pyav():
    assert pyav_adapter.capture_backend_name() == "pyav"


def test_assert_reference_backend_raises_when_pyav_unavailable(monkeypatch):
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(
        pyav_adapter, "PYAV_IMPORT_ERROR", "ImportError: libavdevice.so.62: ..."
    )
    monkeypatch.delenv(pyav_adapter.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        pyav_adapter.assert_reference_backend()

    message = str(exc_info.value)
    # Venv-qualified: the message must never hand anyone a bare `pip`, and
    # this file must not contain one either.
    assert ".venv/bin/pip install -U --only-binary=:all: av" in message
    assert "ImportError: libavdevice.so.62: ..." in message
    assert pyav_adapter.ALLOW_FALLBACK_ENV in message


def test_assert_reference_backend_allows_escape_hatch(monkeypatch):
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(pyav_adapter, "PYAV_IMPORT_ERROR", "ImportError: boom")
    monkeypatch.setenv(pyav_adapter.ALLOW_FALLBACK_ENV, "1")

    assert pyav_adapter.assert_reference_backend() is None


@pytest.mark.parametrize("env_value", [None, "1"])
def test_assert_reference_backend_ok_when_pyav_available(monkeypatch, env_value):
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", True)
    if env_value is None:
        monkeypatch.delenv(pyav_adapter.ALLOW_FALLBACK_ENV, raising=False)
    else:
        monkeypatch.setenv(pyav_adapter.ALLOW_FALLBACK_ENV, env_value)

    assert pyav_adapter.assert_reference_backend() is None


class _NeverReached(Exception):
    """Raised by a stubbed Video to prove the guard ran before any decode."""


def _stub_video(monkeypatch):
    from videocr import api

    def boom(*args, **kwargs):
        raise _NeverReached

    monkeypatch.setattr(api, "Video", boom)
    return api


def test_production_entry_point_refuses_when_pyav_unavailable(monkeypatch):
    """`assert_reference_backend` must have an owner on the real OCR path.

    core/ocr_worker.py -> videocr.api.get_subtitles is the only route a
    pipeline run takes. With the check living solely in tools/fidelity_check
    and a dismissable dialog in main.py, a broken `av` silently demoted every
    run to the non-bit-exact, PTS-estimating fallback.
    """
    api = _stub_video(monkeypatch)
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(pyav_adapter, "PYAV_IMPORT_ERROR", "ImportError: boom")
    monkeypatch.delenv(pyav_adapter.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        api.get_subtitles("no-such-video.mkv")

    assert pyav_adapter.ALLOW_FALLBACK_ENV in str(exc_info.value)


def test_production_entry_point_proceeds_with_escape_hatch(monkeypatch):
    """The opt-in still works: the guard passes and the run gets as far as
    opening the video (here, the stub that stands in for it)."""
    api = _stub_video(monkeypatch)
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(pyav_adapter, "PYAV_IMPORT_ERROR", "ImportError: boom")
    monkeypatch.setenv(pyav_adapter.ALLOW_FALLBACK_ENV, "1")

    with pytest.raises(_NeverReached):
        api.get_subtitles("no-such-video.mkv")


def test_refusal_does_not_leave_a_truncated_ass_file(tmp_path, monkeypatch):
    """save_subtitles_to_file opened 'w+' before doing any work, so anything
    that refused to run -- the backend guard, an unbuildable filter graph --
    truncated or created a zero-byte .ass next to the video. An empty file
    reads as 'OCR produced nothing', not 'OCR did not run'."""
    from videocr import api

    out = tmp_path / "episode.ass"
    out.write_text("previous run's output", encoding="utf-8")

    api_module = _stub_video(monkeypatch)
    monkeypatch.setattr(pyav_adapter, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(pyav_adapter, "PYAV_IMPORT_ERROR", "ImportError: boom")
    monkeypatch.delenv(pyav_adapter.ALLOW_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError):
        api_module.save_subtitles_to_file("no-such-video.mkv", file_path=str(out))

    assert out.read_text(encoding="utf-8") == "previous run's output"

    missing = tmp_path / "fresh.ass"
    with pytest.raises(RuntimeError):
        api.save_subtitles_to_file("no-such-video.mkv", file_path=str(missing))
    assert not missing.exists()
