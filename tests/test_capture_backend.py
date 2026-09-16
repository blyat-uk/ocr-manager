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
    assert "pip install -U --only-binary=:all: av" in message
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
