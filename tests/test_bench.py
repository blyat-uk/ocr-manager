"""Pure tests for the benchmark harness's compare() rendering — no media needed."""
from tools.bench import compare, Measurement


def test_compare_reports_speedup_and_flags_regressions():
    before = {"crop": {"slay": Measurement(seconds=13.3, extra={"probes": 34}).as_dict()}}
    after = {"crop": {"slay": Measurement(seconds=1.05, extra={"probes": 12}).as_dict()}}
    table = compare(before, after)
    assert "12.7x" in table or "12.7×" in table
    assert "slay" in table


def test_compare_marks_a_slowdown():
    before = {"ocr": {"case": Measurement(seconds=10.0).as_dict()}}
    after = {"ocr": {"case": Measurement(seconds=20.0).as_dict()}}
    table = compare(before, after)
    assert "SLOWER" in table.upper()


def test_compare_flags_a_changed_digest_as_fidelity_loss():
    before = {"ocr": {"case": Measurement(seconds=10.0, extra={"digest": "aaa"}).as_dict()}}
    after = {"ocr": {"case": Measurement(seconds=5.0, extra={"digest": "bbb"}).as_dict()}}
    table = compare(before, after)
    assert "FIDELITY" in table.upper()


def test_compare_handles_a_metric_missing_from_before():
    before = {"crop": {}}
    after = {"crop": {"slay": Measurement(seconds=1.0).as_dict()}}
    table = compare(before, after)  # brightness detection has no "before" at all
    assert "new" in table.lower()
