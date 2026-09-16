import threading

import pytest

from videocr import engine_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    engine_registry.reset_registry()
    yield
    engine_registry.reset_registry()


def test_same_arguments_return_the_same_instance(monkeypatch):
    built = []

    def fake_build(lang, det, rec, gpu):
        obj = object()
        built.append(obj)
        return obj

    monkeypatch.setattr(engine_registry, "_build_ocr_engine", fake_build)
    a = engine_registry.get_ocr_engine("ch", None, None, True)
    b = engine_registry.get_ocr_engine("ch", None, None, True)
    assert a is b
    assert len(built) == 1


def test_different_arguments_build_separate_instances(monkeypatch):
    monkeypatch.setattr(engine_registry, "_build_ocr_engine",
                        lambda lang, det, rec, gpu: object())
    a = engine_registry.get_ocr_engine("ch", None, None, True)
    b = engine_registry.get_ocr_engine("en", None, None, True)
    assert a is not b


def test_concurrent_callers_get_one_instance(monkeypatch):
    calls = []

    def slow_build(lang, det, rec, gpu):
        calls.append(1)
        threading.Event().wait(0.05)
        return object()

    monkeypatch.setattr(engine_registry, "_build_ocr_engine", slow_build)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(
            engine_registry.get_ocr_engine("ch", None, None, True)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(map(id, results))) == 1
    assert len(calls) == 1, "engine built more than once under concurrency"


def test_detection_engine_is_cached_separately(monkeypatch):
    monkeypatch.setattr(engine_registry, "_build_ocr_engine",
                        lambda *a: object())
    monkeypatch.setattr(engine_registry, "_build_detection_engine",
                        lambda det, gpu: object())
    ocr = engine_registry.get_ocr_engine("ch", None, None, True)
    det = engine_registry.get_detection_engine(None, True)
    assert ocr is not det
    assert engine_registry.get_detection_engine(None, True) is det
