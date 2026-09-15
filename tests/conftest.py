"""Shared pytest fixtures: synthetic media and the reference-media manifest."""
import json
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


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
