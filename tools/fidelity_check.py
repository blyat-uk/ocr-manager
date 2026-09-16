#!/usr/bin/env python
"""Golden-file fidelity harness.

Runs a fixed set of OCR cases and compares the produced ASS against stored
goldens. Every optimisation must leave these byte-identical; see
docs/superpowers/specs/2026-09-16-ocr-manager-revamp-design.md section 3.

Usage:
    .venv/bin/python tools/fidelity_check.py            # verify
    .venv/bin/python tools/fidelity_check.py --capture  # (re)baseline
    .venv/bin/python tools/fidelity_check.py --case slay_1080p_labels
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CASES_FILE = REPO / "tests" / "fixtures" / "fidelity_cases.json"
MANIFEST_FILE = REPO / "tests" / "fixtures" / "media_manifest.json"
GOLDEN_DIR = REPO / "tests" / "goldens"


@dataclass(frozen=True)
class Case:
    name: str
    project: str
    file: str
    time_ranges: list
    crop: list
    brightness: int
    detect_labels: bool


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    data = json.loads(Path(path).read_text())
    return [Case(**c) for c in data["cases"]]


def media_root(path: Path = MANIFEST_FILE) -> Path:
    return Path(json.loads(Path(path).read_text())["root"])


def digest(ass_text: str) -> str:
    """Hash the Events section only, so header/style changes do not mask
    or fake a difference in recognised text and timing."""
    body = []
    in_events = False
    for line in ass_text.splitlines():
        if line.startswith("[Events]"):
            in_events = True
            continue
        if in_events and line.startswith("Dialogue:"):
            body.append(line.rstrip())
    return hashlib.sha256("\n".join(body).encode("utf-8")).hexdigest()


def run_case(case: Case, root: Path) -> str:
    from videocr.api import get_subtitles
    from videocr.pyav_adapter import assert_reference_backend

    assert_reference_backend()
    video = root / case.project / case.file
    if not video.exists():
        raise FileNotFoundError(video)

    parts = []
    for start, end in case.time_ranges:
        parts.append(get_subtitles(
            str(video), lang="ch", time_start=start, time_end=end,
            conf_threshold=95, sim_threshold=82,
            brightness_threshold=case.brightness,
            similar_image_threshold=0.3, similar_pixel_threshold=25,
            frames_to_skip=0,
            crop_x=case.crop[0], crop_y=case.crop[1],
            crop_width=case.crop[2], crop_height=case.crop[3],
            detect_labels=case.detect_labels,
        ))
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="write goldens instead of verifying")
    ap.add_argument("--case", help="run only this case")
    args = ap.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    root = media_root()
    cases = [c for c in load_cases() if not args.case or c.name == args.case]
    if not cases:
        print("no matching cases", file=sys.stderr)
        return 2

    failures = 0
    for case in cases:
        golden = GOLDEN_DIR / f"{case.name}.ass"
        try:
            ass = run_case(case, root)
        except FileNotFoundError as exc:
            print(f"SKIP {case.name}: missing input {exc}")
            continue

        if args.capture:
            golden.write_text(ass, encoding="utf-8")
            print(f"WROTE {case.name} ({digest(ass)[:12]})")
            continue

        if not golden.exists():
            print(f"MISSING GOLDEN {case.name} — run with --capture")
            failures += 1
            continue

        want = digest(golden.read_text(encoding="utf-8"))
        got = digest(ass)
        if want == got:
            print(f"OK   {case.name} ({got[:12]})")
        else:
            failures += 1
            print(f"FAIL {case.name}: golden {want[:12]} != produced {got[:12]}")
            (GOLDEN_DIR / f"{case.name}.actual.ass").write_text(ass, encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
