from videocr import pyav_adapter


def test_pyav_is_available():
    assert pyav_adapter.PYAV_AVAILABLE, (
        f"PyAV failed to import: {pyav_adapter.PYAV_IMPORT_ERROR}"
    )


def test_backend_is_pyav():
    assert pyav_adapter.capture_backend_name() == "pyav"
