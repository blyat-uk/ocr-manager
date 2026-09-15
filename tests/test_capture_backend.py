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
