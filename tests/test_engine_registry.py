"""The engine pool hands each caller an engine nobody else is using.

PaddleOCR / TextDetection instances are not safe to call from two threads at
once (measured: 4 threads on one TextDetection changed 101-126 of 384
outputs, and 4 threads on one PaddleOCR misread 17 texts and killed a
thread). These tests pin the lease contract that prevents that: a leased
engine is exclusive to its lessee, an idle one is reused rather than
rebuilt, and a lease always comes back -- also when its body raises.
"""
import threading

import pytest

from videocr import engine_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    engine_registry.reset_registry()
    yield
    engine_registry.reset_registry()


class _Builds:
    """A builder that records every engine it makes."""

    def __init__(self, delay=0.0):
        self.engines = []
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, *args):
        if self.delay:
            threading.Event().wait(self.delay)
        engine = object()
        with self._lock:
            self.engines.append(engine)
        return engine


def test_sequential_leases_of_one_key_reuse_one_instance(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", builds)

    seen = []
    for _ in range(3):
        with engine_registry.lease_ocr_engine("ch", None, None, True) as engine:
            seen.append(engine)

    assert len(builds.engines) == 1
    assert seen[0] is seen[1] is seen[2] is builds.engines[0]


def test_concurrent_leases_get_distinct_instances(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", builds)

    concurrency = 6

    def burst():
        all_holding = threading.Barrier(concurrency, timeout=10)
        held = []
        errors = []

        def lessee():
            try:
                with engine_registry.lease_ocr_engine("ch", None, None, True) as engine:
                    held.append(engine)
                    # Every lessee holds its lease until all of them do, so no
                    # engine can have been returned and handed on in between.
                    all_holding.wait()
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=lessee) for _ in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        return held

    first = burst()
    assert len(set(map(id, first))) == concurrency
    assert len(builds.engines) == concurrency

    # Once all of them are back in the pool, a second burst of the same size
    # reuses them -- and still gets one distinct instance per lessee.
    second = burst()
    assert len(set(map(id, second))) == concurrency
    assert set(map(id, second)) == set(map(id, builds.engines))
    assert len(builds.engines) == concurrency


def test_a_lease_is_returned_when_its_body_raises(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(engine_registry, "_build_detection_engine", builds)

    with pytest.raises(RuntimeError):
        with engine_registry.lease_detection_engine(None, True) as engine:
            first = engine
            raise RuntimeError("inference failed")

    with engine_registry.lease_detection_engine(None, True) as engine:
        assert engine is first
    assert len(builds.engines) == 1


def test_a_nested_lease_of_the_same_key_gets_a_different_instance(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", builds)

    with engine_registry.lease_ocr_engine("ch", None, None, True) as outer:
        with engine_registry.lease_ocr_engine("ch", None, None, True) as inner:
            assert inner is not outer
    assert len(builds.engines) == 2


def test_different_arguments_never_share_an_instance(monkeypatch):
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", _Builds())
    with engine_registry.lease_ocr_engine("ch", None, None, True) as a:
        pass
    with engine_registry.lease_ocr_engine("en", None, None, True) as b:
        pass
    assert a is not b


def test_detection_and_ocr_engines_are_pooled_separately(monkeypatch):
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", _Builds())
    monkeypatch.setattr(engine_registry, "_build_detection_engine", _Builds())
    with engine_registry.lease_ocr_engine("ch", None, None, True) as ocr:
        pass
    with engine_registry.lease_detection_engine(None, True) as det:
        pass
    assert ocr is not det
    with engine_registry.lease_detection_engine(None, True) as again:
        assert again is det


def test_construction_never_runs_twice_at_once(monkeypatch):
    """Construction is serialised process-wide: `utils.suppress_output()`
    redirects the process's stdout/stderr file descriptors, and two builds
    overlapping in it left fd 1 and fd 2 on /dev/null for good (measured,
    3 of 3 runs). Inference is not serialised -- see the distinct-instances
    test above, whose lessees all hold engines at the same time."""
    ocr_builds = _Builds(delay=0.02)
    det_builds = _Builds(delay=0.02)
    in_build = {"now": 0, "max": 0}
    lock = threading.Lock()

    def tracking(builder):
        def build(*args):
            with lock:
                in_build["now"] += 1
                in_build["max"] = max(in_build["max"], in_build["now"])
            try:
                return builder(*args)
            finally:
                with lock:
                    in_build["now"] -= 1
        return build

    monkeypatch.setattr(engine_registry, "_build_ocr_engine", tracking(ocr_builds))
    monkeypatch.setattr(engine_registry, "_build_detection_engine", tracking(det_builds))

    all_holding = threading.Barrier(8, timeout=10)

    def lessee(i):
        lease = (engine_registry.lease_ocr_engine("ch", None, None, True) if i % 2
                 else engine_registry.lease_detection_engine(None, True))
        with lease:
            all_holding.wait()

    threads = [threading.Thread(target=lessee, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ocr_builds.engines) == 4 and len(det_builds.engines) == 4
    assert in_build["max"] == 1


def test_a_failed_build_leaves_nothing_behind(monkeypatch):
    calls = []

    def failing(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("no GPU")
        return object()

    monkeypatch.setattr(engine_registry, "_build_ocr_engine", failing)
    with pytest.raises(RuntimeError):
        with engine_registry.lease_ocr_engine("ch", None, None, True):
            pass  # pragma: no cover - never entered

    with engine_registry.lease_ocr_engine("ch", None, None, True) as engine:
        assert engine is not None
    assert len(calls) == 2


def test_an_engine_leased_before_a_reset_is_not_pooled_after_it(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", builds)
    with engine_registry.lease_ocr_engine("ch", None, None, True) as stale:
        engine_registry.reset_registry()
    with engine_registry.lease_ocr_engine("ch", None, None, True) as fresh:
        assert fresh is not stale
    assert len(builds.engines) == 2


def test_the_shared_instance_getters_are_gone():
    """Handing one engine to every caller is what corrupted concurrent OCR;
    nothing may be able to reach that API again."""
    for name in ("get_ocr_engine", "get_detection_engine", "engine_lock"):
        assert not hasattr(engine_registry, name), name
