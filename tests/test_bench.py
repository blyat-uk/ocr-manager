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


# --- What the suites count (M4) -----------------------------------------------


def test_ocr_instrumentation_counts_frames_decoded_by_grab_as_well_as_read(synthetic_video):
    """Label phase 1 decodes most frames with grab() and converts only the
    sampled ones with read(); "frames decoded" must count both."""
    from tools.bench import _install_ocr_instrumentation
    from videocr import pyav_adapter

    stats, restore = _install_ocr_instrumentation()
    try:
        with pyav_adapter.Capture(str(synthetic_video)) as cap:
            assert cap.read()[0]
            assert cap.grab()
            assert cap.grab()
            assert cap.read()[0]
    finally:
        restore()
    assert stats["frames_decoded"] == 4


def test_crop_suite_reports_the_probes_detection_used(monkeypatch, synthetic_video, tmp_path):
    """The crop detector fetches probes through its own fetch layer, never
    Capture.read(), so counting reads always reported 0: the count must be
    the detector's own CropResult.probes_used."""
    from core.detect import crop as crop_module
    from core.detect.crop import CropResult
    from tools import bench

    monkeypatch.setattr("videocr.utils.create_detection_engine", lambda det_model_dir, use_gpu: object())
    calls = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None, settings=None, cancel_check=None):
        calls.append((video_path, duration_sec, det_engine, consensus, settings, cancel_check))
        return CropResult(box=(10, 200, 300, 30), sample_pts=[0.1] * 13, hit_pts=[0.1], agreed=5,
                          probes_used=13, flagged="low-agreement", frame_size=(320, 240))

    monkeypatch.setattr(crop_module, "detect_crop", fake_detect_crop)

    entry = {"video": synthetic_video, "dir": tmp_path}
    result = bench._run_crop_case("synthetic", entry)

    assert result is not None
    assert result["extra"]["probes"] == 13
    assert result["extra"]["detected"] == {"x": 10, "y": 200, "width": 300, "height": 30}
    # A box withheld from the UI for review is still a measured detection.
    assert result["extra"]["flagged"] == "low-agreement"
    assert result["extra"]["auto_applicable"] is False
    # Detection runs on an engine leased from the registry, never a shared one.
    (_path, _duration, engine, consensus, settings, _cancel), = calls
    assert engine is not None and consensus == [] and settings is None
