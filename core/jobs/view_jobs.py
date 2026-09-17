"""Frame and strip jobs: the pixels the review views draw (plan 3C).

Two sources, never mixed -- see the frame-addressing table in
core/detect/__init__.py:

    FrameJob   core.detect.crop.grab_frames(band_frac=1.0): the whole frame,
               scaled to at most FRAME_HEIGHT rows. The frame returned for a
               time is the first one at or after that time rounded to whole
               ms, so re-fetching the same time always gives the same frame.
               These are NOT the pixels the OCR pass sees (a different filter
               order, and on some sources a different frame altogether): the
               crop canvas and its filmstrip draw them, nothing that previews
               or tunes a brightness threshold may.

    StripJob   core.detect.ocr_view.grab_ocr_strips_at: the crop strip of the
               frame at index round(t * fps), pixel-identical to what the OCR
               pass hands its brightness filter. Everything that shows OCR
               pixels -- the brightness tiles, any masked preview -- uses
               these, and only for the crop box they were grabbed with, which
               is why the box travels with the result.

Both are Qt-free, run on the CPU lane and need no engine lease (pixels only,
no inference). Their results carry numpy arrays; app/imaging.py is the only
place those become QImages.

Identity
    key       f"frames:{file}:{hash(times)}" / f"strips:{file}:{crop_box}:{hash(times)}".
              A request for other times (or another box) is a different job,
              so several can be in flight for one file; an identical request
              replaces a queued one (the runner's rule), which is exactly
              what a view repainting wants.
    priority  0, like the auto-pilot's own jobs.

Cancellation follows core.jobs.detect_jobs' convention: a job that saw the
request returns None rather than a partial result. Neither frame source takes
a cancel_check, so the request is read before the grab and again after it;
a grab already under way runs to its end (one batch of frames).

Failed grabs are dropped, never raised: a time missing from the result is a
time the view has no pixels for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.detect import crop as _crop
from core.detect import ocr_view as _ocr_view
from core.jobs.runner import JobContext, Lane

if TYPE_CHECKING:
    import numpy as np

FRAME_HEIGHT = 720        # rows a canvas frame is scaled to at most (grab_frames never upscales)


@dataclass(frozen=True)
class FramesResult:
    file: str
    frames: dict[float, np.ndarray]      # requested time -> whole BGR frame; a failed grab has no entry


@dataclass(frozen=True)
class StripsResult:
    file: str
    crop_box: tuple[int, int, int, int]  # the box the strips were grabbed with: they are only valid for it
    strips: dict[float, np.ndarray]      # requested time -> OCR-exact, unmasked BGR strip


class FrameJob:
    """Whole frames of one file at `times`, for the crop canvas and filmstrip.

    `target_height` is an upper bound: grab_frames scales to
    min(target_height, the source's height), never up. It is deliberately not
    part of the key, and the controller never passes another one: a file's
    frames exist once, at FRAME_HEIGHT, and a view that draws them smaller
    scales what it was given.
    """

    kind = "frames"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, file: str, times: list[float], target_height: int = FRAME_HEIGHT):
        self.file = file
        self.times = tuple(float(time) for time in times)
        self.key = f"{self.kind}:{file}:{hash(self.times)}"
        self.video_path = os.path.join(project_dir, file)
        self.target_height = int(target_height)

    def run(self, ctx: JobContext) -> FramesResult | None:
        if ctx.cancelled():
            return None
        if not self.times:
            return FramesResult(self.file, {})
        frames = self._grab(self.times)
        if ctx.cancelled():
            return None
        return FramesResult(self.file, frames)

    def _grab(self, times: tuple[float, ...]) -> dict[float, np.ndarray]:
        """`times` paired with their frames, without the ones that failed.

        grab_frames returns frames only, so a dropped grab shifts every later
        frame: when the batch comes back short, the times are re-fetched one
        at a time rather than pair a frame with the wrong time. Repeated
        times share one grab (the same frame, by construction).
        """
        wanted = tuple(dict.fromkeys(times))
        frames = _crop.grab_frames(self.video_path, list(wanted), band_frac=1.0,
                                   target_height=self.target_height)
        if len(frames) == len(wanted):
            return dict(zip(wanted, frames))
        grabbed = {}
        for time in wanted:
            one = _crop.grab_frames(self.video_path, [time], band_frac=1.0,
                                    target_height=self.target_height)
            if one:
                grabbed[time] = one[0]
        return grabbed


class StripJob:
    """OCR-exact crop strips of one file at `times`, for the brightness tiles
    and every other preview of what the OCR pass reads."""

    kind = "strips"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, file: str, crop_box: tuple[int, int, int, int], times: list[float]):
        self.file = file
        self.crop_box = tuple(int(value) for value in crop_box)
        self.times = tuple(float(time) for time in times)
        self.key = f"{self.kind}:{file}:{self.crop_box}:{hash(self.times)}"
        self.video_path = os.path.join(project_dir, file)

    def run(self, ctx: JobContext) -> StripsResult | None:
        if ctx.cancelled():
            return None
        if not self.times:
            return StripsResult(self.file, self.crop_box, {})
        # grab_ocr_strips_at returns (requested time, strip) pairs, so a
        # dropped time simply has no pair: nothing to re-align here.
        pairs = _ocr_view.grab_ocr_strips_at(self.video_path, self.crop_box,
                                             list(dict.fromkeys(self.times)))
        if ctx.cancelled():
            return None
        return StripsResult(self.file, self.crop_box, {float(time): strip for time, strip in pairs})
