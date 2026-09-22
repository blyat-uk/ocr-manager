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

The disk sits in front of both of them
    Decoding a frame costs a seek and a decode of a 1080p source; reading one
    back out of core.jobs.view_cache costs a WebP read. So every job here is
    disk-first: it asks the file's FileViewCache for each time it was given,
    sends only the misses to the frame source, and writes each image it did
    decode back. Coming back to a file the user has already looked at --
    after a restart, after walking the queue -- then costs no decoding at
    all, and WarmJob pays the first visit's price ahead of time.

    The cache is a cache, never a dependency: a read that misses, a project
    that cannot be written to, a directory wiped because the video changed
    all degrade to decoding everything, which is what these jobs always did.

Lossy frames, lossless strips
    Frames are stored lossily (a canvas to look at, ~84 KB against 571 KB),
    so where a frame came from matters: FramesResult.lossy names the times
    that came off the disk, and everything else in it is the decoder's own
    pixels. A caller that measures levels rather than looks at them -- the
    crop view's masked preview, whose threshold a lossy round trip moves by
    up to ~58 -- asks with `exact=True`, which skips the read entirely and
    decodes the lot. It still writes what it decoded back: storing it lossily
    is no loss to the next caller that is only looking, and `exact` is part
    of the key so an exact request is never answered by a queued cached one.

    Strips are stored losslessly, because the Brightness tab measures on them
    and a lossy round trip would move the threshold it proposes. A cached
    strip is therefore the strip the OCR pass reads, byte for byte: read hits
    and fresh decodes are equally exact, and StripsResult has no provenance
    to report.

Warming
    WarmJob decodes one file's whole `view_cache.Wanted` set -- its crop
    evidence's frames, its brightness tiles' and gallery lines' strips -- into
    the cache and keeps none of it: the point is the disk, so the result
    carries counts, not arrays. It shares the fetch helpers below, so a time
    already on disk is never decoded twice, and it finishes by trimming
    whatever `keep` no longer names (a re-detect's old sample times, the
    strips of a crop box that has moved).

Identity
    key       f"frames:{file}:{hash(times)}:{exact|cached}" /
              f"strips:{file}:{crop_box}:{hash(times)}" / f"warm:{file}".
              A request for other times (or another box) is a different job,
              so several can be in flight for one file; an identical request
              replaces a queued one (the runner's rule), which is exactly
              what a view repainting wants. Warming is keyed by the file
              alone: one pass per file, and a newer `keep` replaces a queued
              older one.
    priority  VIEW_PRIORITY: above every priority AutoPilot gives its CPU
              work (at most 15), because someone is looking at these pixels
              and the CPU lane has two workers for a whole folder's metadata,
              thumbnails and audio profiles; below the proof's 100, which the
              user asked for explicitly.
              WARM_PRIORITY: -1, below every priority AutoPilot gives (its
              lowest is ranges at 0), because nobody is waiting on a warm
              pass. It may only have a CPU worker no detection job wants, and
              AutoPilot holds it outright (`is_held_job`) while a run is
              active or the user paused, so warming never competes with the
              run's own CPU decoding.

Cancellation follows core.jobs.detect_jobs' convention: a job that saw the
request returns None rather than a partial result. Neither frame source takes
a cancel_check, so the request is read before the grab and again after it;
a grab already under way runs to its end (one batch of frames). A warm pass
decodes in batches of WARM_BATCH so that the gap between two of them comes
round often enough for a run to take the machine promptly; the frames it did
warm stay on disk, because a cache is allowed to be half full -- only its
*result* is None.

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
from core.jobs.view_cache import FileViewCache, Wanted

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np

    from core.jobs.view_cache import Box

FRAME_HEIGHT = 720        # rows a canvas frame is scaled to at most (grab_frames never upscales)
VIEW_PRIORITY = 50        # between AutoPilot's CPU work (<= 15) and the proof (100)
WARM_PRIORITY = -1        # below every AutoPilot priority (ranges is 0): warming yields to all detection
WARM_BATCH = 4            # times per decode call while warming: how often a warm pass can notice a cancel


@dataclass(frozen=True)
class FramesResult:
    file: str
    frames: dict[float, np.ndarray]      # requested time -> whole BGR frame; a failed grab has no entry
    # The times whose frame came from the lossy disk cache rather than the
    # decoder. Everything else in `frames` is exact, whether or not it was
    # then written to the cache lossily. The crop view's "masked" preview
    # filters on real pixel levels, so it may only use exact frames.
    lossy: frozenset[float] = frozenset()


@dataclass(frozen=True)
class StripsResult:
    file: str
    crop_box: tuple[int, int, int, int]  # the box the strips were grabbed with: they are only valid for it
    strips: dict[float, np.ndarray]      # requested time -> OCR-exact, unmasked BGR strip


@dataclass(frozen=True)
class WarmResult:
    """What one file's warm pass did, in counts.

    Nothing here is a picture: a folder's worth of these may sit in the event
    queue at once, and the pixels they are about belong to the disk.
    `frames_written` and `strips_written` are what the cache took (a project
    that cannot be written to reports zeroes, having decoded them all for
    nothing), `trimmed` how many stale images went.
    """

    file: str
    frames_written: int
    strips_written: int
    trimmed: int


# --------------------------------------------------------------------------
# Fetch-or-decode, shared by all three jobs
# --------------------------------------------------------------------------

def _grab_frames(video_path: str, times: Sequence[float], target_height: int) -> dict[float, np.ndarray]:
    """`times` paired with their whole frames, without the ones that failed.

    grab_frames returns frames only, so a dropped grab shifts every later
    frame: when the batch comes back short, the times are re-fetched one at a
    time rather than pair a frame with the wrong time. Repeated times share
    one grab (the same frame, by construction).
    """
    wanted = tuple(dict.fromkeys(times))
    frames = _crop.grab_frames(video_path, list(wanted), band_frac=1.0, target_height=target_height)
    if len(frames) == len(wanted):
        return dict(zip(wanted, frames))
    grabbed = {}
    for time in wanted:
        one = _crop.grab_frames(video_path, [time], band_frac=1.0, target_height=target_height)
        if one:
            grabbed[time] = one[0]
    return grabbed


def _grab_strips(video_path: str, crop_box: Box, times: Sequence[float]) -> dict[float, np.ndarray]:
    """`times` paired with their OCR-exact strips, without the ones that
    failed. grab_ocr_strips_at returns (requested time, strip) pairs, so a
    dropped time simply has no pair: nothing to re-align here."""
    pairs = _ocr_view.grab_ocr_strips_at(video_path, crop_box, list(dict.fromkeys(times)))
    return {float(time): strip for time, strip in pairs}


def _batches(times: Sequence[float], size: int | None) -> tuple[tuple[float, ...], ...]:
    """`times` in chunks of at most `size`, or in one chunk when `size` is
    None -- a view asks for a handful of times and one grab opens the video
    once, so there is nothing to chop up there. No times, no chunks: a decode
    call is never made for nothing."""
    if not times:
        return ()
    if size is None:
        return (tuple(times),)
    return tuple(tuple(times[start:start + size]) for start in range(0, len(times), size))


def _cached(times: Iterable[float], read: Callable[[float], np.ndarray | None] | None) -> dict[float, np.ndarray]:
    """The times `read` answered, paired with its pixels, in request order.

    `read` is None for an exact request: the disk holds a lossy copy of these
    frames and the caller has said it cannot use one, so it is not consulted
    at all."""
    hits: dict[float, np.ndarray] = {}
    if read is None:
        return hits
    for time in times:
        image = read(time)
        if image is not None:
            hits[time] = image
    return hits


def _missing(times: Iterable[float], read: Callable[[float], np.ndarray | None]) -> tuple[float, ...]:
    """The times `read` has no image for, de-duplicated in request order.

    Warming wants to know that a time is on disk, not what it looks like, so
    a hit's pixels are dropped where they are read -- this is the one caller
    that must not end up holding a file's worth of frames."""
    return tuple(time for time in dict.fromkeys(times) if read(time) is None)


def _discard(time: float, image: np.ndarray) -> None:
    """`_decode_into`'s `keep` for a warm pass: the image went to the disk and
    goes nowhere else."""


def _decode_into(times: Sequence[float],
                 decode: Callable[[tuple[float, ...]], dict[float, np.ndarray]],
                 write: Callable[[float, np.ndarray], bool],
                 keep: Callable[[float, np.ndarray], None],
                 *, batch: int | None,
                 cancelled: Callable[[], bool]) -> int:
    """Decode `times`, store each image and hand it to `keep`; return how many
    the cache took.

    The cache is written to first and asked about nothing: a write that fails
    (a read-only project) is not worth a word to the caller beyond the count,
    and `keep` gets the image either way. `batch` bounds one decode call --
    see `_batches`. Cancellation is read between batches only: neither frame
    source takes a cancel_check, so a batch under way always runs to its end,
    and what it decoded is stored rather than thrown away.
    """
    written = 0
    for chunk in _batches(times, batch):
        if cancelled():
            break
        for time, image in decode(chunk).items():
            if write(time, image):
                written += 1
            keep(time, image)
    return written


def _fetch(times: Sequence[float],
           read: Callable[[float], np.ndarray | None] | None,
           decode: Callable[[tuple[float, ...]], dict[float, np.ndarray]],
           write: Callable[[float, np.ndarray], bool],
           cancelled: Callable[[], bool]) -> tuple[dict[float, np.ndarray], frozenset[float]]:
    """The images for `times` -- off the disk where they are, out of the
    decoder where they are not -- and the times the disk answered.

    The two halves of what a view job does, in the order that matters: read
    first, decode only what is left, and put every fresh image back so the
    next request for it is a read. A time neither the disk nor the decoder
    could produce is simply absent, as it always was.
    """
    wanted = tuple(dict.fromkeys(times))
    images = _cached(wanted, read)
    hits = frozenset(images)                 # taken before the decoder adds its own to `images`
    _decode_into(tuple(time for time in wanted if time not in hits),
                 decode, write, images.__setitem__, batch=None, cancelled=cancelled)
    return images, hits


# --------------------------------------------------------------------------
# The jobs
# --------------------------------------------------------------------------

class WarmJob:
    """Decode one file's view pixels into `core.jobs.view_cache` and throw
    them away.

    The point is the disk, not the return value, so a warm result carries
    counts rather than arrays and nothing here holds an image beyond the
    write that stores it: the queue warms a whole folder, and a 432-file
    folder may not become a memory spike.

    It shares FrameJob's and StripJob's fetch helpers, so a time already on
    disk is never decoded twice, and finishes by trimming whatever `keep` does
    not name. At WARM_PRIORITY it sits below every detection job, and
    AutoPilot holds it (`is_held_job`) while a run is active or the user
    paused, so warming never competes with a run's own CPU decoding.
    """

    kind = "warm"
    lane = Lane.CPU
    priority = WARM_PRIORITY

    def __init__(self, project_dir: str, file: str, keep: Wanted):
        self.file = file
        self.keep = keep
        self.key = f"{self.kind}:{file}"
        self.project_dir = project_dir
        self.video_path = os.path.join(project_dir, file)

    def run(self, ctx: JobContext) -> WarmResult | None:
        if ctx.cancelled():
            return None
        cache = FileViewCache(self.project_dir, self.file, self.video_path)
        frames_written = _decode_into(
            _missing(self.keep.frame_times, cache.read_frame),
            lambda times: _grab_frames(self.video_path, times, FRAME_HEIGHT),
            cache.write_frame, _discard, batch=WARM_BATCH, cancelled=ctx.cancelled)

        box = self.keep.crop_box
        strips_written = 0
        # No box, no strips: they are keyed by it, and a file without a crop
        # has none to hold.
        if box is not None and not ctx.cancelled():
            strips_written = _decode_into(
                _missing(self.keep.strip_times, lambda time: cache.read_strip(box, time)),
                lambda times: _grab_strips(self.video_path, box, times),
                lambda time, image: cache.write_strip(box, time, image),
                _discard, batch=WARM_BATCH, cancelled=ctx.cancelled)

        # Tidying up is the least urgent thing a warm pass does, so a cancel
        # leaves it for the next one: what is on disk is right either way,
        # there is just more of it than `keep` asked for.
        if ctx.cancelled():
            return None
        return WarmResult(self.file, frames_written, strips_written, cache.trim(self.keep))


class FrameJob:
    """Whole frames of one file at `times`, for the crop canvas and filmstrip.

    `target_height` is an upper bound: grab_frames scales to
    min(target_height, the source's height), never up. It is deliberately not
    part of the key, and the controller never passes another one: a file's
    frames exist once, at FRAME_HEIGHT, and a view that draws them smaller
    scales what it was given.

    `exact` refuses the disk's lossy copies: the request goes straight to the
    decoder, every frame it answers with is the decoder's own (`lossy` comes
    back empty) -- and is still written to the cache, because the callers who
    only look at these frames are glad of it.
    """

    kind = "frames"
    lane = Lane.CPU
    priority = VIEW_PRIORITY

    def __init__(self, project_dir: str, file: str, times: list[float], target_height: int = FRAME_HEIGHT,
                 exact: bool = False):
        self.file = file
        self.times = tuple(float(time) for time in times)
        # `exact` is part of the key: an exact request must not be answered by
        # a queued lossy one (JobRunner replaces an identical key).
        self.key = f"{self.kind}:{file}:{hash(self.times)}:{'exact' if exact else 'cached'}"
        self.project_dir = project_dir
        self.video_path = os.path.join(project_dir, file)
        self.target_height = int(target_height)
        self.exact = bool(exact)

    def run(self, ctx: JobContext) -> FramesResult | None:
        if ctx.cancelled():
            return None
        if not self.times:
            return FramesResult(self.file, {})
        cache = FileViewCache(self.project_dir, self.file, self.video_path)
        frames, hits = _fetch(self.times,
                              None if self.exact else cache.read_frame,
                              lambda times: _grab_frames(self.video_path, times, self.target_height),
                              cache.write_frame,
                              ctx.cancelled)
        if ctx.cancelled():
            return None
        return FramesResult(self.file, frames, lossy=hits)


class StripJob:
    """OCR-exact crop strips of one file at `times`, for the brightness tiles
    and every other preview of what the OCR pass reads.

    The cache stores these losslessly, so a strip off the disk is the strip
    the decoder would have produced: there is no exact/lossy distinction to
    make here, and none in the result.
    """

    kind = "strips"
    lane = Lane.CPU
    priority = VIEW_PRIORITY

    def __init__(self, project_dir: str, file: str, crop_box: tuple[int, int, int, int], times: list[float]):
        self.file = file
        self.crop_box = tuple(int(value) for value in crop_box)
        self.times = tuple(float(time) for time in times)
        self.key = f"{self.kind}:{file}:{self.crop_box}:{hash(self.times)}"
        self.project_dir = project_dir
        self.video_path = os.path.join(project_dir, file)

    def run(self, ctx: JobContext) -> StripsResult | None:
        if ctx.cancelled():
            return None
        if not self.times:
            return StripsResult(self.file, self.crop_box, {})
        cache = FileViewCache(self.project_dir, self.file, self.video_path)
        strips, _ = _fetch(self.times,
                           lambda time: cache.read_strip(self.crop_box, time),
                           lambda times: _grab_strips(self.video_path, self.crop_box, times),
                           lambda time, image: cache.write_strip(self.crop_box, time, image),
                           ctx.cancelled)
        if ctx.cancelled():
            return None
        return StripsResult(self.file, self.crop_box, strips)
