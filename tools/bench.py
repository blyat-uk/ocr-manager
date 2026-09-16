"""Benchmark harness for Stage 2 before/after comparisons.

Measures three suites against the reference media in `/mnt/FAST/work`
(see `tests/fixtures/media_manifest.json`):

  ocr    - the three `tools/fidelity_check.py` cases plus one 4K case,
           split into model-load / decode / OCR-inference / label-scan time.
  crop   - `core/subtitle_detector.py`'s SubtitleDetectionWorker, one file
           per reference project.
  ranges - `core/audio_analysis.py` + `core/audio_finder/`, cold (no cache)
           and warm (cache present), one project run per project.

Every suite skips cleanly when its media is absent, exactly like
`tools/fidelity_check.py`, and none of them ever write into
`/mnt/FAST/work/*` -- OCR/crop runs only read video files, and the ranges
suite always points `db_path` at a temp directory instead of the project's
own `.audio_fingerprints.db`.

Usage:
    .venv/bin/python tools/bench.py --suite all --label NAME --repeat 2 \\
        --out docs/superpowers/stage2-records/baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.fidelity_check import Case, load_cases, media_root, run_case, digest  # noqa: E402

VENV_PYTHON = REPO / ".venv" / "bin" / "python"
MANIFEST_FILE = REPO / "tests" / "fixtures" / "media_manifest.json"

# One extra 4K case, from a reference project not used by the fidelity
# goldens. Crop/brightness come from the user's own hand-verified
# `.ocr.json` for this project (tests/fixtures/detector_truth.json does not
# exist yet -- it is created in a later Stage 2 task). Kept to a short clip
# (2 minutes, inside the project's verified dialogue range) to bound
# runtime; the 4K source is downscaled to 1080p at decode time same as any
# other >1080p source, so this exercises that path specifically.
XWZ_4K_CASE = Case(
    name="xwz_4k_dialogue",
    project="XWZ",
    file="XWZ_-_169_[4K]10BHQ.mp4",
    time_ranges=[["2:20", "4:20"]],
    crop=[576, 1892, 2688, 108],
    brightness=230,
    detect_labels=False,
)


# --------------------------------------------------------------------------
# Measurement / compare()
# --------------------------------------------------------------------------

@dataclass
class Measurement:
    seconds: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"seconds": self.seconds, "extra": dict(self.extra)}

    @staticmethod
    def from_dict(d: dict) -> "Measurement":
        return Measurement(seconds=d["seconds"], extra=dict(d.get("extra", {})))


def _fmt_seconds(s: float) -> str:
    return f"{s:.2f}s"


def _fmt_ratio(before_s: float, after_s: float) -> str:
    """Render before/after as a change cell. `before_s / after_s > 1` means
    `after` was faster."""
    if before_s <= 0 or after_s <= 0:
        return "n/a"
    ratio = before_s / after_s
    if ratio >= 1.0:
        return f"{ratio:.1f}x faster"
    return f"{1.0 / ratio:.2f}x SLOWER"


def compare(before: dict, after: dict) -> str:
    """Render a markdown before/after table.

    `before` and `after` are `{suite: {key: Measurement.as_dict()}}` trees
    (exactly what one suite run produces, NOT the full baseline-file
    envelope with label/git_sha/timestamp -- callers pass `data["suites"]`
    for that). A `compare({}, after)` call (nothing in `before`) renders a
    solo summary of `after`, since every metric is then "new" -- this is how
    the CLI prints a just-captured baseline with no prior run to diff
    against.
    """
    lines = ["| suite · metric | before | after | change |", "| --- | --- | --- | --- |"]

    suites = sorted(set(before) | set(after))
    for suite in suites:
        before_suite = before.get(suite) or {}
        after_suite = after.get(suite) or {}
        keys = sorted(set(before_suite) | set(after_suite))

        for key in keys:
            label = f"{suite} · {key}"
            b_raw = before_suite.get(key)
            a_raw = after_suite.get(key)

            if a_raw is None:
                b = Measurement.from_dict(b_raw)
                lines.append(f"| {label} | {_fmt_seconds(b.seconds)} | — | removed |")
                continue

            a = Measurement.from_dict(a_raw)

            if b_raw is None:
                lines.append(f"| {label} | — | {_fmt_seconds(a.seconds)} | new |")
                continue

            b = Measurement.from_dict(b_raw)

            b_digest = b.extra.get("digest")
            a_digest = a.extra.get("digest")
            if b_digest is not None and a_digest is not None and b_digest != a_digest:
                change = f"FIDELITY CHANGED ({b_digest[:8]} -> {a_digest[:8]})"
            else:
                change = _fmt_ratio(b.seconds, a.seconds)

            lines.append(
                f"| {label} | {_fmt_seconds(b.seconds)} | {_fmt_seconds(a.seconds)} | {change} |"
            )

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reference-media resolution (mirrors tests/conftest.py's reference_media(),
# reimplemented without pytest so this can skip cleanly from a plain CLI).
# --------------------------------------------------------------------------

def _resolve_reference_projects() -> dict:
    """Returns {key: {"dir": Path, "video": Path, "crop": [...], "brightness": int}}
    for every manifest project whose directory/video actually exists.
    Empty dict (never an exception) when the media root is absent."""
    manifest = json.loads(MANIFEST_FILE.read_text())
    root = Path(manifest["root"])
    if not root.exists():
        return {}

    resolved = {}
    for key, entry in manifest["projects"].items():
        project_dir = root / entry["dir"]
        if not project_dir.is_dir():
            continue
        if entry["file"]:
            video = project_dir / entry["file"]
        else:
            candidates = sorted(
                p for p in project_dir.iterdir()
                if p.suffix.lower() in (".mkv", ".mp4")
            )
            video = candidates[0] if candidates else None
        if video is None or not video.exists():
            continue
        resolved[key] = {
            "dir": project_dir,
            "video": video,
            "crop": entry["crop"],
            "brightness": entry["brightness"],
        }
    return resolved


def _ocr_json_crop(project_dir: Path, filename: str) -> dict | None:
    """The user's hand-verified crop for one file, straight from the
    project's own `.ocr.json` -- used as ground truth for the crop suite
    since `tests/fixtures/detector_truth.json` does not exist yet."""
    ocr_json = project_dir / ".ocr.json"
    if not ocr_json.exists():
        return None
    try:
        data = json.loads(ocr_json.read_text())
        return data.get("files", {}).get(filename, {}).get("crop")
    except (json.JSONDecodeError, OSError):
        return None


def _median_by_seconds(runs: list[dict]) -> dict:
    """Pick one whole run dict as "the median" of `--repeat N` runs.

    For odd N this is the literal middle sample. For even N (notably the
    brief's --repeat 2) there is no single middle *record* -- only a middle
    *value" -- and averaging fields like frame counts or digests across two
    different runs would produce a Measurement that no run actually
    produced. So this always returns an observed run, using the
    upper-median index (n // 2), which for N=2 is the SLOWER of the two --
    a deliberate conservative choice, noted in the report.
    """
    runs_sorted = sorted(runs, key=lambda r: r["seconds"])
    return runs_sorted[len(runs_sorted) // 2]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# OCR suite
# --------------------------------------------------------------------------

def _install_ocr_instrumentation():
    """Monkeypatch model construction, Capture.read, ocr.predict, and the
    label scanner's scan() to time model-load / decode / OCR-inference /
    label-scan, and count frames decoded / frames sent to OCR / "extend"
    queue messages. Same style the brief calls for (monkeypatch Capture.read,
    ocr.predict, engine construction) extended to the label-scan phase and
    frame/message counts, which needed their own seams.

    Returns (stats_dict, restore_callable). Intended to be installed once
    inside a fresh, dedicated subprocess per OCR case run (see
    `_run_ocr_case_in_subprocess`) so timings/counts/peak-RSS are never
    contaminated by a previous case or by model weights already resident
    from an earlier run in the same process.
    """
    import queue as queue_mod
    from videocr import utils as vc_utils
    from videocr import pyav_adapter as vc_pyav
    from videocr import label_scanner as vc_label

    stats = {
        "model_load_s": 0.0,
        "decode_s": 0.0,
        "ocr_inference_s": 0.0,
        "label_scan_s": 0.0,
        "frames_decoded": 0,
        "frames_sent_ocr": 0,
        "extend_count": 0,
    }

    orig_create_ocr = vc_utils.create_ocr_engine
    orig_create_det = vc_utils.create_detection_engine
    orig_capture_read = vc_pyav.Capture.read
    orig_scan = vc_label.LabelScanner.scan
    orig_queue_put = queue_mod.Queue.put

    def timed_create_ocr(*a, **kw):
        t0 = time.perf_counter()
        engine = orig_create_ocr(*a, **kw)
        stats["model_load_s"] += time.perf_counter() - t0

        orig_predict = engine.predict

        def timed_predict(batch, *pa, **pk):
            t1 = time.perf_counter()
            # Force eager evaluation here (not at the call site's own
            # list()) so the timer captures real inference time even if
            # predict() itself is lazy/generator-based.
            result = list(orig_predict(batch, *pa, **pk))
            stats["ocr_inference_s"] += time.perf_counter() - t1
            try:
                stats["frames_sent_ocr"] += len(batch)
            except TypeError:
                stats["frames_sent_ocr"] += 1
            return result

        engine.predict = timed_predict
        return engine

    def timed_create_det(*a, **kw):
        t0 = time.perf_counter()
        engine = orig_create_det(*a, **kw)
        stats["model_load_s"] += time.perf_counter() - t0
        return engine

    def timed_read(self, *a, **kw):
        t0 = time.perf_counter()
        result = orig_capture_read(self, *a, **kw)
        stats["decode_s"] += time.perf_counter() - t0
        ret = result[0] if isinstance(result, tuple) else result
        if ret:
            stats["frames_decoded"] += 1
        return result

    def timed_scan(self, *a, **kw):
        t0 = time.perf_counter()
        result = orig_scan(self, *a, **kw)
        stats["label_scan_s"] += time.perf_counter() - t0
        return result

    def counting_put(self, item, *a, **kw):
        if isinstance(item, tuple) and item and item[0] == "extend":
            stats["extend_count"] += 1
        return orig_queue_put(self, item, *a, **kw)

    vc_utils.create_ocr_engine = timed_create_ocr
    vc_utils.create_detection_engine = timed_create_det
    vc_pyav.Capture.read = timed_read
    vc_label.LabelScanner.scan = timed_scan
    queue_mod.Queue.put = counting_put

    def restore():
        vc_utils.create_ocr_engine = orig_create_ocr
        vc_utils.create_detection_engine = orig_create_det
        vc_pyav.Capture.read = orig_capture_read
        vc_label.LabelScanner.scan = orig_scan
        queue_mod.Queue.put = orig_queue_put

    return stats, restore


def _ocr_worker_main(case_json: Path, out_json: Path) -> None:
    """Runs inside a dedicated subprocess: one OCR case, once."""
    import resource

    spec = json.loads(case_json.read_text())
    case = Case(**spec)
    root = media_root()

    stats, restore = _install_ocr_instrumentation()
    try:
        t0 = time.perf_counter()
        try:
            ass = run_case(case, root)
        except FileNotFoundError as exc:
            out_json.write_text(json.dumps({"skip": True, "reason": str(exc)}))
            return
        total_s = time.perf_counter() - t0
    finally:
        restore()

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    accounted = stats["model_load_s"] + stats["decode_s"] + stats["ocr_inference_s"] + stats["label_scan_s"]
    other_s = max(0.0, total_s - accounted)

    result = Measurement(
        seconds=total_s,
        extra={
            "digest": digest(ass),
            "model_load_s": stats["model_load_s"],
            "decode_s": stats["decode_s"],
            "ocr_inference_s": stats["ocr_inference_s"],
            "label_scan_s": stats["label_scan_s"],
            "other_s": other_s,
            "frames_decoded": stats["frames_decoded"],
            "frames_sent_ocr": stats["frames_sent_ocr"],
            "extend_count": stats["extend_count"],
            "peak_rss_kb": peak_rss_kb,
        },
    ).as_dict()
    out_json.write_text(json.dumps(result))


def _run_ocr_case_in_subprocess(case: Case) -> dict:
    with tempfile.TemporaryDirectory(prefix="ocr-bench-") as td:
        case_json = Path(td) / "case.json"
        out_json = Path(td) / "result.json"
        case_json.write_text(json.dumps(asdict(case)))

        proc = subprocess.run(
            [str(VENV_PYTHON), str(Path(__file__).resolve()),
             "--_ocr-worker-json", str(case_json),
             "--_ocr-worker-out", str(out_json)],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if not out_json.exists():
            raise RuntimeError(
                f"OCR bench worker for {case.name} produced no output "
                f"(exit {proc.returncode}). stderr:\n{proc.stderr[-4000:]}"
            )
        return json.loads(out_json.read_text())


def run_ocr_suite(repeat: int) -> dict:
    if not media_root().exists():
        print(f"SKIP ocr: media root {media_root()} not present", file=sys.stderr)
        return {}

    cases = list(load_cases()) + [XWZ_4K_CASE]
    out = {}
    for case in cases:
        video_path = media_root() / case.project / case.file
        if not video_path.exists():
            print(f"SKIP ocr/{case.name}: missing {video_path}", file=sys.stderr)
            continue

        runs = []
        for i in range(repeat):
            result = _run_ocr_case_in_subprocess(case)
            if result.get("skip"):
                print(f"SKIP ocr/{case.name}: {result.get('reason')}", file=sys.stderr)
                runs = []
                break
            runs.append(result)
            print(f"  ocr/{case.name} run {i + 1}/{repeat}: {result['seconds']:.2f}s", file=sys.stderr)

        if runs:
            out[case.name] = _median_by_seconds(runs)

    return out


# --------------------------------------------------------------------------
# Crop suite
# --------------------------------------------------------------------------

def _probe_duration_seconds(video: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


def _run_crop_case(key: str, entry: dict) -> dict | None:
    """Drives SubtitleDetectionWorker synchronously (no QThread) for one
    file, timing internals directly rather than spinning a Qt event loop --
    `_run()` is fully synchronous already, and pyqtSignal delivers a direct
    (same-thread) connection's slot call immediately with no queued/event
    -loop dispatch needed."""
    from PyQt6.QtCore import QCoreApplication
    from core.subtitle_detector import SubtitleDetectionWorker
    from videocr import pyav_adapter as vc_pyav

    QCoreApplication.instance() or QCoreApplication([])

    video = entry["video"]
    duration = _probe_duration_seconds(video)

    probe_count = {"n": 0}
    orig_read = vc_pyav.Capture.read

    def counting_read(self, *a, **kw):
        result = orig_read(self, *a, **kw)
        ret = result[0] if isinstance(result, tuple) else result
        if ret:
            probe_count["n"] += 1
        return result

    detected = {}

    def on_detected(filename, slider_pos, cx, cy, cw, ch):
        detected[filename] = (cx, cy, cw, ch)

    vc_pyav.Capture.read = counting_read
    try:
        worker = SubtitleDetectionWorker([(video.name, str(video), duration)])
        worker.file_detected.connect(on_detected)
        t0 = time.perf_counter()
        worker._run()
        elapsed = time.perf_counter() - t0
    finally:
        vc_pyav.Capture.read = orig_read

    if video.name not in detected:
        print(f"SKIP crop/{key}: detector did not resolve a crop for {video.name}", file=sys.stderr)
        return None

    cx, cy, cw, ch = detected[video.name]
    extra = {
        "probes": probe_count["n"],
        "detected": {"x": cx, "y": cy, "width": cw, "height": ch},
    }

    truth = _ocr_json_crop(entry["dir"], video.name)
    if truth is not None:
        extra["truth"] = truth
        extra["dx"] = cx - truth["x"]
        extra["dy"] = cy - truth["y"]
        extra["dw"] = cw - truth["width"]
        extra["dh"] = ch - truth["height"]
    else:
        extra["truth"] = None  # no .ocr.json crop to compare against; recorded for later

    return Measurement(seconds=elapsed, extra=extra).as_dict()


def run_crop_suite(repeat: int) -> dict:
    projects = _resolve_reference_projects()
    if not projects:
        print("SKIP crop: no reference projects resolved from manifest", file=sys.stderr)
        return {}

    out = {}
    for key, entry in projects.items():
        runs = []
        for i in range(repeat):
            m = _run_crop_case(key, entry)
            if m is None:
                runs = []
                break
            runs.append(m)
            print(f"  crop/{key} run {i + 1}/{repeat}: {m['seconds']:.2f}s", file=sys.stderr)
        if runs:
            out[key] = _median_by_seconds(runs)

    return out


# --------------------------------------------------------------------------
# Ranges suite
# --------------------------------------------------------------------------

def _install_ranges_instrumentation():
    import core.audio_analysis as vc_audio
    from core.audio_finder.audio import pipeline as vc_pipeline

    stats = {"identity_s": 0.0, "decode_s": 0.0, "fingerprint_s": 0.0}

    orig_sha256 = vc_audio.compute_sha256
    orig_extract = vc_pipeline.extract_audio
    orig_spectrogram = vc_pipeline.compute_spectrogram
    orig_peaks = vc_pipeline.find_peaks
    orig_hashes = vc_pipeline.generate_fingerprints

    def timed(bucket, fn):
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            result = fn(*a, **kw)
            stats[bucket] += time.perf_counter() - t0
            return result
        return wrapped

    vc_audio.compute_sha256 = timed("identity_s", orig_sha256)
    vc_pipeline.extract_audio = timed("decode_s", orig_extract)
    vc_pipeline.compute_spectrogram = timed("fingerprint_s", orig_spectrogram)
    vc_pipeline.find_peaks = timed("fingerprint_s", orig_peaks)
    vc_pipeline.generate_fingerprints = timed("fingerprint_s", orig_hashes)

    def restore():
        vc_audio.compute_sha256 = orig_sha256
        vc_pipeline.extract_audio = orig_extract
        vc_pipeline.compute_spectrogram = orig_spectrogram
        vc_pipeline.find_peaks = orig_peaks
        vc_pipeline.generate_fingerprints = orig_hashes

    return stats, restore


def _project_video_files(project_dir: Path) -> list[str]:
    return sorted(
        p.name for p in project_dir.iterdir()
        if p.suffix.lower() in (".mkv", ".mp4")
    )


def _run_ranges_pass(project_dir: Path, filenames: list[str], db_path: Path) -> dict:
    """One ingest+analyze+compute pass against `db_path` (never inside
    `project_dir` -- the whole point of calling _ingest/_analyze/
    _compute_time_ranges directly instead of the worker's _run(), which
    hardcodes db_path = project_dir/.audio_fingerprints.db)."""
    from core.audio_analysis import AudioAnalysisWorker, DEFAULT_MIN_SEGMENT_SEC
    from core.audio_finder.config import AnalysisConfig, MatchConfig

    stats, restore = _install_ranges_instrumentation()
    try:
        worker = AudioAnalysisWorker(str(project_dir), filenames)
        cfg = AnalysisConfig(match=MatchConfig(min_length_sec=DEFAULT_MIN_SEGMENT_SEC), db_path=str(db_path))

        t0 = time.perf_counter()
        tag_id, profile_id = worker._ingest(cfg, "bench")
        ingest_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        worker._analyze(cfg, tag_id, profile_id)
        match_s = time.perf_counter() - t0

        results = worker._compute_time_ranges(cfg, tag_id, profile_id)
    finally:
        restore()

    total_s = ingest_s + match_s
    return {
        "seconds": total_s,
        "identity_s": stats["identity_s"],
        "decode_s": stats["decode_s"],
        "fingerprint_s": stats["fingerprint_s"],
        "match_s": match_s,
        "keep_ranges": results,
    }


def _run_ranges_project(key: str, entry: dict) -> tuple[dict | None, dict | None]:
    project_dir = entry["dir"]
    filenames = _project_video_files(project_dir)
    if len(filenames) < 2:
        print(f"SKIP ranges/{key}: needs >=2 video files to match against, found {len(filenames)}",
              file=sys.stderr)
        return None, None

    with tempfile.TemporaryDirectory(prefix="ranges-bench-") as td:
        db_path = Path(td) / "bench_fingerprints.db"
        cold = _run_ranges_pass(project_dir, filenames, db_path)
        warm = _run_ranges_pass(project_dir, filenames, db_path)

    if cold["keep_ranges"] != warm["keep_ranges"]:
        print(f"WARNING ranges/{key}: cold and warm keep_ranges differ -- "
              f"cold={cold['keep_ranges']} warm={warm['keep_ranges']}", file=sys.stderr)

    cold_m = Measurement(seconds=cold["seconds"], extra={
        "identity_s": cold["identity_s"], "decode_s": cold["decode_s"],
        "fingerprint_s": cold["fingerprint_s"], "match_s": cold["match_s"],
        "keep_ranges": cold["keep_ranges"],
    }).as_dict()
    warm_m = Measurement(seconds=warm["seconds"], extra={
        "identity_s": warm["identity_s"], "decode_s": warm["decode_s"],
        "fingerprint_s": warm["fingerprint_s"], "match_s": warm["match_s"],
        "keep_ranges": warm["keep_ranges"],
    }).as_dict()
    return cold_m, warm_m


def run_ranges_suite(repeat: int) -> dict:
    projects = _resolve_reference_projects()
    if not projects:
        print("SKIP ranges: no reference projects resolved from manifest", file=sys.stderr)
        return {}

    out = {}
    for key, entry in projects.items():
        cold_runs, warm_runs = [], []
        for i in range(repeat):
            cold_m, warm_m = _run_ranges_project(key, entry)
            if cold_m is None:
                cold_runs, warm_runs = [], []
                break
            cold_runs.append(cold_m)
            warm_runs.append(warm_m)
            print(f"  ranges/{key} run {i + 1}/{repeat}: cold={cold_m['seconds']:.2f}s "
                  f"warm={warm_m['seconds']:.2f}s", file=sys.stderr)
        if cold_runs:
            out[f"{key}_cold"] = _median_by_seconds(cold_runs)
            out[f"{key}_warm"] = _median_by_seconds(warm_runs)

    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SUITE_RUNNERS = {
    "ocr": run_ocr_suite,
    "crop": run_crop_suite,
    "ranges": run_ranges_suite,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["ocr", "crop", "ranges", "all"], default="all")
    ap.add_argument("--out", type=Path, default=None, help="write the results JSON here")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--repeat", type=int, default=1)
    # Hidden subprocess-worker mode (see _run_ocr_case_in_subprocess). Not
    # part of the public CLI surface described in the task brief.
    ap.add_argument("--_ocr-worker-json", type=Path, default=None)
    ap.add_argument("--_ocr-worker-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args._ocr_worker_json is not None:
        _ocr_worker_main(args._ocr_worker_json, args._ocr_worker_out)
        return 0

    suite_names = ["ocr", "crop", "ranges"] if args.suite == "all" else [args.suite]

    suites = {}
    for name in suite_names:
        print(f"=== running suite: {name} (repeat={args.repeat}) ===", file=sys.stderr)
        suites[name] = SUITE_RUNNERS[name](args.repeat)

    envelope = {
        "label": args.label,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "machine shared with a concurrent GPU workload; absolute numbers are noisy, treat as relative baselines",
        "repeat": args.repeat,
        "suites": suites,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(envelope, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)

    print(compare({}, suites))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
