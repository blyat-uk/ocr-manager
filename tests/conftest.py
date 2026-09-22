"""Shared pytest fixtures: synthetic media and the reference-media manifest."""
import json
import subprocess
from pathlib import Path

import pytest

from videocr import engine_registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_engine_registry():
    """Every test starts and ends with an empty OCR engine pool.

    Without this, a test that monkeypatches create_ocr_engine /
    create_detection_engine expecting a fresh build per call can silently
    lease an idle engine pooled by an earlier test that happened to use
    the same (lang, det, rec, gpu) / (det, gpu) key -- see
    tests/test_candidate_buffer.py's own in-test reset for the intra-test
    version of this same hazard.
    """
    engine_registry.reset_registry()
    yield
    engine_registry.reset_registry()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """10 frames, 320x240, 25 fps, yuv420p H.264 in MP4."""
    out = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=0.4",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "5", str(out),
    ])
    return out


@pytest.fixture(scope="session")
def offset_video(tmp_path_factory) -> Path:
    """Same as synthetic_video, but muxed with -output_ts_offset so the
    container's start_time (and every frame's PTS) is offset by 1.5s.

    This exercises the container start-time path that a fresh,
    edit-list-free MP4 (like synthetic_video) cannot: its start_time is
    genuinely 0, so a test built only on it can't tell a correct
    start-time implementation from a hardcoded `return 0.0`.
    """
    out = tmp_path_factory.mktemp("media") / "offset.mp4"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=0.4",
        "-output_ts_offset", "1.5",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "5", str(out),
    ])
    # Verify the mechanism actually produced a non-zero container
    # start_time on this ffmpeg build before any test relies on it.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )
    start_time = float(json.loads(probe.stdout)["format"]["start_time"])
    assert start_time == pytest.approx(1.5, abs=1e-3), (
        f"-output_ts_offset did not produce the expected container "
        f"start_time on this ffmpeg build (got {start_time}); "
        "the offset_video fixture needs a different mechanism here."
    )
    return out


@pytest.fixture(scope="session")
def synthetic_subtitle_video(tmp_path_factory) -> Path:
    """75 frames, 640x360, 25 fps. A white bar sits in the subtitle band
    for frames 25-49 only, so gating and timing can be asserted exactly."""
    out = tmp_path_factory.mktemp("media") / "subs.mp4"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#202020:size=640x360:rate=25:duration=3",
        "-vf", "drawbox=x=160:y=300:w=320:h=30:color=white@1.0:t=fill:"
               "enable='between(n,25,49)'",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "25", str(out),
    ])
    return out


@pytest.fixture(scope="session")
def delayed_video_subtitle_clip(tmp_path_factory) -> Path:
    """synthetic_subtitle_video's picture (75 frames, the white bar on frames
    25-49) muxed into MKV with its first frame at 0.080 s, while the audio and
    the container start at 0 -- the layout of Jinwu Guard episodes 07, 08,
    09, 12 and 15. No B-frames, so no muxer shift moves the audio too."""
    out = tmp_path_factory.mktemp("media") / "delayed.mkv"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-itsoffset", "0.08",
        "-f", "lavfi", "-i", "color=c=#202020:size=640x360:rate=25:duration=3,"
                             "drawbox=x=160:y=300:w=320:h=30:color=white@1.0:t=fill:"
                             "enable='between(n,25,49)'",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3.2",
        "-map", "0:v", "-map", "1:a",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-bf", "0", "-g", "25",
        "-c:a", "pcm_s16le", str(out),
    ])
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=start_time:stream=codec_type,start_time", "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout)
    starts = {s["codec_type"]: float(s["start_time"]) for s in probe["streams"]}
    assert float(probe["format"]["start_time"]) == 0.0, probe
    assert starts == {"video": pytest.approx(0.08, abs=1e-6), "audio": 0.0}, probe
    return out


@pytest.fixture(scope="session")
def synthetic_audio_video(tmp_path_factory) -> Path:
    """10 s, 320x240, with 1 kHz tone bursts at 1-3 s, 5-6 s and 8-9 s."""
    out = tmp_path_factory.mktemp("media") / "audio.mp4"
    bursts = "+".join([
        "between(t,1,3)", "between(t,5,6)", "between(t,8,9)",
    ])
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=10",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration=10",
        "-filter_complex", f"[1:a]volume='if({bursts},1,0)':eval=frame[a]",
        "-map", "0:v", "-map", "[a]",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", str(out),
    ])
    return out


@pytest.fixture(scope="session")
def video_only_clip(tmp_path_factory) -> Path:
    """20 s, 640x360, 25 fps, NO audio stream. A white bar sits in the
    bottom band during the first second of every two, so uniform probing
    over the 40-60% window (8-12 s) hits it on some probes and not others
    (which keeps it from looking like a static watermark)."""
    out = tmp_path_factory.mktemp("media") / "video_only.mp4"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#202020:size=640x360:rate=25:duration=20",
        "-vf", "drawbox=x=120:y=300:w=400:h=30:color=white@1.0:t=fill:"
               "enable='lt(mod(t,2),1)'",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "25", str(out),
    ])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.split() == ["video"], probe.stdout
    return out


@pytest.fixture(scope="session")
def undecodable_audio_clip(tmp_path_factory) -> Path:
    """A Matroska file whose audio stream EXISTS but cannot be decoded: a
    PCM track with its codec ID overwritten by an unknown one of the same
    length. ffmpeg fails to extract audio from it exactly as it does for a
    file with no audio stream at all (exit 234), so only probing for the
    stream tells the two apart."""
    media = tmp_path_factory.mktemp("media")
    pcm = media / "pcm.mkv"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=4",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "pcm_s16le", str(pcm),
    ])
    data = pcm.read_bytes()
    assert data.count(b"A_PCM/INT/LIT") == 1
    out = media / "undecodable_audio.mkv"
    out.write_bytes(data.replace(b"A_PCM/INT/LIT", b"A_QQQ/QQQ/QQQ"))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.split() == ["video", "audio"], probe.stdout
    return out


@pytest.fixture(scope="session")
def reference_media() -> dict:
    """Reference projects, or skip the test when they are not present."""
    manifest = json.loads((FIXTURES / "media_manifest.json").read_text())
    root = Path(manifest["root"])
    if not root.exists():
        pytest.skip(f"reference media root {root} not present")

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
            "duration": _ffprobe_duration(video),
        }

    if not resolved:
        pytest.skip("no reference projects resolved from manifest")
    return resolved


@pytest.fixture(scope="session")
def detector_truth() -> dict:
    """Ground truth crop/brightness values extracted from each reference
    project's own .ocr.json (see tests/fixtures/detector_truth.json and
    task-2-brief.md Step 1). Skips when the fixture file is absent."""
    truth_path = FIXTURES / "detector_truth.json"
    if not truth_path.exists():
        pytest.skip(f"{truth_path} not present")
    return json.loads(truth_path.read_text())
