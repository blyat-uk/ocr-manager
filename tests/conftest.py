"""Shared pytest fixtures: synthetic media and the reference-media manifest."""
import json
import subprocess
from pathlib import Path

import pytest

from videocr import engine_registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_engine_registry():
    """Every test starts and ends with an empty OCR engine registry.

    Without this, a test that monkeypatches create_ocr_engine /
    create_detection_engine expecting a fresh build per call can silently
    get back a stale engine cached by an earlier test that happened to use
    the same (lang, det, rec, gpu) / (det, gpu) key -- see
    tests/test_candidate_buffer.py's own in-test reset for the intra-test
    version of this same hazard.
    """
    engine_registry.reset_registry()
    yield
    engine_registry.reset_registry()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


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
        }

    if not resolved:
        pytest.skip("no reference projects resolved from manifest")
    return resolved
