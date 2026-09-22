"""The on-disk store for the pixels the review views draw (plan 3C).

Opening a file's Crop or Brightness tab costs a seek and a decode per time --
around 350 ms a frame on a 1080p episode, and the same frames come back every
time the file is selected again. The pixels themselves are small: written as
WebP, the whole 720-row canvas frame is about 84 KB at quality 90 (571 KB
lossless, for a picture nobody measures anything on) and an OCR strip about
30 KB lossless, and either decodes back in about 4.4 ms. So the cache trades
a few hundred KB a file for two orders of magnitude on the wait, and
`core.jobs.view_jobs` asks it before it asks a decoder.

Layout, one directory per video under the project's cache, named by the same
name digest `core.project.store.evidence_path` uses (a name of any shape
hashes to one short, filesystem-safe directory name):

    .ocr-cache/view/<sha256 of the file name>/
        meta.json                          {"version", "size", "mtime_ns"} of the video
        f-000120.000.webp                  whole frame, lossy (FRAME_QUALITY)
        t-000120.000.webp                  queue thumbnail, lossless, 72 rows
        s-288_786_1344_53-000120.000.webp  OCR-exact strip, lossless, keyed by the crop box

Lossy frames, lossless strips
    The two kinds of pixels are not worth the same. A frame is a canvas to
    look at: the crop box is dragged over it, and a WebP artefact in the
    picture behind the box changes nothing anybody measures -- 84 KB instead
    of 571 KB is the right trade. A strip is evidence: the Brightness tab
    reads its threshold off those exact levels, and a lossy round trip would
    move the number the user is choosing. Strips therefore go through WebP's
    lossless mode (cv2 quality 101) and come back bit-exact;
    tests/test_view_cache.py pins that on noise and on a subtitle-like
    image. Because a frame is not exact, `FramesResult.lossy` tells the view
    which of its frames came from here, and anything that filters on real
    pixel levels uses only the exact ones.

    A strip is only valid for the crop box it was cut with, so the box is
    part of its name rather than of its content: a strip for another box is a
    miss, never the old pixels.

Naming a time
    A file name holds the time to the millisecond, zero-padded to a fixed
    width (`%010.3f`) so that a listing of the directory reads in time order.
    Evidence times are whole milliseconds in practice, and a time this format
    cannot spell exactly is not cached at all: the alternative -- rounding
    1/3 s onto its neighbour's file -- would hand a view the pixels of a
    different moment, which is worse than decoding it again. So every key is
    formatted and parsed straight back, and a time that does not survive that
    round trip misses and is never written.

Identity, and everything that can go wrong
    A cached frame of a video that has since been replaced would show the
    wrong picture, silently, with nothing on screen to say so. `meta.json`
    carries the video's (size, mtime_ns); opening the cache checks it and
    wipes every image in the directory when it does not match, when it is
    missing, or when it is junk. If the video cannot be stat'd at all there
    is nothing to check against, so the cache switches itself off (`readable`
    is False): every read misses and every write is a no-op.

    Beyond that the rule is that a cache may cost a decode but must never
    fail a job. A read-only mount, a full disk, a directory somebody chmod'ed
    away: every write swallows OSError and returns False, and every read
    returns None rather than raise -- including for a truncated or corrupt
    WebP, which decodes to None. Writes land with tmp + fsync + os.replace,
    the same way `core.project.store` writes `.ocr.json`, because a reader
    that picked up a half-written WebP would draw a torn frame; a failed
    write takes its temporary file with it.

Bounded by what is on screen
    The cache has no size limit and needs none, because it only ever holds
    what a file's tabs draw: `FileViewCache.trim` drops the times a file's
    evidence no longer names and the strips of a crop box that has moved, and
    `prune` drops the whole directory of a video that has left the folder. In
    a folder worked through and cleared out, the steady state is exactly the
    pixels the tabs would draw today.

Qt-free, like everything under core/: numpy arrays in, numpy arrays out, and
`app/imaging.py` is the only place they become QImages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from core.project.store import CACHE_DIRNAME

if TYPE_CHECKING:
    from core.project.model import FileEntry

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
VIEW_DIRNAME = "view"
FRAME_QUALITY = 90        # cv2 WebP quality for whole frames; strips ignore this and go lossless
LOSSLESS_QUALITY = 101    # cv2's WebP lossless mode: > 100 means "lossless", the strips' bit-exact round trip
META_FILENAME = "meta.json"

TIME_FORMAT = "%010.3f"   # 000120.000: milliseconds, fixed width, so a listing sorts by time
_FRAME_NAME = re.compile(r"^f-(-?\d+\.\d{3})\.webp$")
_STRIP_NAME = re.compile(r"^s-(-?\d+_-?\d+_-?\d+_-?\d+)-(-?\d+\.\d{3})\.webp$")
_THUMB_NAME = re.compile(r"^t-(-?\d+\.\d{3})\.webp$")
_IMAGE_NAME = re.compile(r"^[fst]-.*\.webp$")
_DIGEST_NAME = re.compile(r"^[0-9a-f]{64}$")   # a directory view_dir named, and nothing else

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Wanted:
    """The times a file's tabs actually draw: what the cache should hold and
    nothing else (`FileViewCache.trim` deletes the rest)."""

    frame_times: tuple[float, ...]
    crop_box: Box | None
    strip_times: tuple[float, ...]
    thumbnail_time: float | None = None

    def __bool__(self) -> bool:
        """False when there is nothing to warm: a file with no evidence yet is
        not worth a warm job.

        `thumbnail_time` deliberately does not count. It says which thumbnail
        `trim` must keep, not work for a warm pass to do -- the thumbnail is
        ThumbnailJob's, written when the queue first needed it -- so a file
        whose evidence names nothing would otherwise earn a job with nothing
        to decode.
        """
        return bool(self.frame_times) or bool(self.crop_box and self.strip_times)


def _times(values) -> list[float]:
    """The floats in `values`, skipping anything that is not a number.

    Evidence is a disposable cache that may come back partial or junk (the
    same rule the views read it by), so a malformed entry is dropped rather
    than raising."""
    times = []
    for value in values or ():
        try:
            times.append(float(value))
        except (TypeError, ValueError):
            continue
    return times


def wanted(entry: FileEntry, thumbnail_time: float | None = None) -> Wanted:
    """The frame and strip times `entry`'s Crop and Brightness tabs draw.

    Frames: the crop evidence's sample times, de-duplicated in order -- the
    Crop tab fetches exactly `read_samples(evidence["crop"])`'s times, and
    they are the only times whose frames re-fetch the ones the evidence was
    measured on (the frame-addressing table in core/detect/__init__.py).

    Strips: the Brightness tab's zoom tiles (`evidence["brightness"]["tiles"]`,
    a label -> time mapping) plus the gallery's lines
    (`evidence["lines"]["samples"]`), on the file's current crop box. Strips
    are keyed by that box, so a file with no crop has none to hold.

    `thumbnail_time` is the queue thumbnail's, which is not in evidence at
    all: which frame a file's row shows is AutoPilot's policy (`sample_time`,
    else THUMBNAIL_FRACTION of the duration), so the caller that knows that
    rule passes it in rather than have this module import the scheduler it is
    imported by. Left out, the file simply has no thumbnail to hold -- and
    `trim` would then drop one it has.

    Evidence is a disposable cache: every key is read with `.get`, and a
    missing or malformed one simply contributes no times.
    """
    evidence = entry.evidence or {}
    crop_evidence = evidence.get("crop") or {}
    samples = crop_evidence.get("samples") or ()
    frame_times = _times(sample.get("time") for sample in samples if isinstance(sample, dict))

    box = None if entry.crop is None else (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height)
    brightness_evidence = evidence.get("brightness") or {}
    tiles = brightness_evidence.get("tiles") or {}
    strip_times = _times(tiles.values() if isinstance(tiles, dict) else ())
    lines = evidence.get("lines") or {}
    line_samples = lines.get("samples") or ()
    strip_times += _times(sample.get("time") for sample in line_samples if isinstance(sample, dict))

    return Wanted(frame_times=tuple(dict.fromkeys(frame_times)),
                  thumbnail_time=None if thumbnail_time is None else float(thumbnail_time),
                  crop_box=box,
                  strip_times=tuple(dict.fromkeys(strip_times)))


def _name_digest(file: str) -> str:
    """The directory name a video's view pixels live under: the sha256 of the
    name's bytes, as `core.project.store.evidence_path` hashes them. `prune`
    decides what to keep by the same digest, so the two cannot drift."""
    return hashlib.sha256(os.fsencode(file)).hexdigest()


def view_dir(project_dir: str, file: str) -> Path:
    """`<project_dir>/.ocr-cache/view/<sha256 hex of the file name>`, the same
    name-digest convention as `core.project.store.evidence_path`."""
    return Path(project_dir) / CACHE_DIRNAME / VIEW_DIRNAME / _name_digest(file)


def prune(project_dir: str, names: Iterable[str]) -> int:
    """Delete the view-cache directory of every video no longer in the folder;
    how many went.

    `FileViewCache.trim` tidies within one video's directory, which is no help
    to a video that has been deleted or renamed away: its directory is named
    after a file nothing will ask for again, so nothing would ever revisit it.
    Without this the cache would grow forever in a folder that is worked
    through and cleared out, which is the one thing its "hold exactly what the
    tabs draw" rule is meant to prevent.

    Only directories named the way `view_dir` names them are touched -- 64
    lowercase hex characters -- so a stray file or a directory somebody else
    put there is left alone, the way `core.project.store` guards its evidence
    sweep. Nothing raises: an absent or unreadable cache prunes nothing, and a
    directory that will not go is logged and counted out.
    """
    directory = Path(project_dir) / CACHE_DIRNAME / VIEW_DIRNAME
    keep = {_name_digest(name) for name in names}
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return 0
    except OSError as exc:
        logger.debug("View cache could not list %s (%s)", directory, exc)
        return 0
    pruned = 0
    for item in entries:
        if item.name in keep or not _DIGEST_NAME.match(item.name):
            continue
        try:
            if not item.is_dir():
                continue                                         # a loose file is not ours to remove
            shutil.rmtree(item.path)
        except OSError as exc:
            logger.debug("View cache could not remove %s (%s)", item.path, exc)
            continue
        pruned += 1
    return pruned


def _time_key(time: float) -> str | None:
    """`time` as a file name holds it, or None when the name could not hold
    it exactly.

    The key is parsed straight back and compared: a time that does not
    survive (anything finer than a millisecond, a NaN, an infinity) has no
    file rather than share the file of a time a millisecond away.
    """
    try:
        value = float(time)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    key = TIME_FORMAT % value
    try:
        return key if float(key) == value else None
    except ValueError:                                           # pragma: no cover - %f always parses
        return None


def _box_key(box: Box | None) -> str | None:
    """`box` as a file name holds it (`x_y_w_h`), or None when it is not four
    numbers."""
    if box is None:
        return None
    try:
        x, y, width, height = (int(value) for value in box)
    except (TypeError, ValueError):
        return None
    return f"{x}_{y}_{width}_{height}"


def _frame_name(time_key: str) -> str:
    return f"f-{time_key}.webp"


def _strip_name(box_key: str, time_key: str) -> str:
    return f"s-{box_key}-{time_key}.webp"


def _thumb_name(time_key: str) -> str:
    return f"t-{time_key}.webp"


def _atomic_write(path: Path, data: bytes) -> bool:
    """`data` onto `path` atomically: a unique temporary file beside it,
    fsynced, then os.replace()d on. False (and no leftovers) when anything
    fails -- see the module docstring: the cache never raises into a job."""
    tmp = None
    try:
        handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("View cache could not write %s (%s)", path, exc)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _encode(image, params: list[int]) -> bytes | None:
    """`image` as WebP bytes, or None when OpenCV will not encode it (an
    empty array, a dtype it has no WebP for): not something to fail a job
    over."""
    if image is None or getattr(image, "size", 0) == 0:
        return None
    try:
        ok, buffer = cv2.imencode(".webp", image, params)
    except cv2.error as exc:
        logger.debug("View cache could not encode an image (%s)", exc)
        return None
    return buffer.tobytes() if ok else None


class FileViewCache:
    """One video's view pixels on disk.

    Constructing it validates `meta.json` against the video's (size,
    mtime_ns): a mismatch wipes the directory, because a cached frame of a
    video that has been replaced would silently show the wrong picture.
    `readable` is False when the video cannot be stat'd at all, and every
    read then misses and every write is a no-op.

    Every write swallows OSError and returns False: a project directory that
    cannot be written to (a read-only mount, a full disk) must degrade to
    today's decode-every-time behaviour, never fail a job.
    """

    def __init__(self, project_dir: str, file: str, video_path: str):
        self.file = file
        self.video_path = video_path
        self.directory = view_dir(project_dir, file)
        self._meta: dict | None = None                           # the meta the pixels here belong to
        self._meta_on_disk = False                               # ... and whether it has been written
        self._readable = self._open()

    # -- identity ---------------------------------------------------------

    def _open(self) -> bool:
        """Check the directory against the video, wiping it when it does not
        belong to it. False when the cache cannot be trusted at all."""
        try:
            stat = os.stat(self.video_path)
        except OSError as exc:
            logger.debug("View cache has no source to check against: %s (%s)", self.video_path, exc)
            return False
        self._meta = {"version": FORMAT_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if self._read_meta() == self._meta:
            self._meta_on_disk = True
            return True
        if not self.directory.exists():                          # nothing cached: meta follows the first write
            return True
        # These pixels came from another video (or from nothing we can
        # identify). Refusing to serve them is only safe if they are gone:
        # a wipe we could not finish switches the cache off instead.
        if not self._wipe():
            return False
        self._ensure_meta()               # the emptied directory now says which video it is for
        return True

    def _read_meta(self) -> dict | None:
        try:
            data = json.loads((self.directory / META_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _wipe(self) -> bool:
        """Delete every cached image in the directory (meta.json is rewritten,
        not deleted). False when one of them would not go."""
        try:
            names = [entry.name for entry in os.scandir(self.directory)]
        except OSError:
            return False
        for name in names:
            if not _IMAGE_NAME.match(name):
                continue
            try:
                os.unlink(self.directory / name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.debug("View cache could not wipe %s (%s)", name, exc)
                return False
        return True

    def _ensure_meta(self) -> bool:
        """The directory exists and holds this video's meta.json. Images are
        only ever written next to a meta that names their source."""
        if self._meta_on_disk:
            return True
        if self._meta is None:
            return False
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("View cache could not create %s (%s)", self.directory, exc)
            return False
        text = json.dumps(self._meta, separators=(",", ":"))
        self._meta_on_disk = _atomic_write(self.directory / META_FILENAME, text.encode("utf-8"))
        return self._meta_on_disk

    @property
    def readable(self) -> bool:
        return self._readable

    # -- pixels -----------------------------------------------------------

    def read_frame(self, time: float) -> np.ndarray | None:
        """The cached whole frame at `time` as BGR uint8, or None for a miss
        (not cached, unreadable file, corrupt image)."""
        key = _time_key(time)
        return None if key is None else self._read(_frame_name(key))

    def write_frame(self, time: float, image: np.ndarray) -> bool:
        """Store `image` as the whole frame at `time`; True when it landed."""
        key = _time_key(time)
        if key is None:
            return False
        return self._write(_frame_name(key), image, [int(cv2.IMWRITE_WEBP_QUALITY), FRAME_QUALITY])

    def read_thumbnail(self, time: float) -> np.ndarray | None:
        """The cached queue thumbnail at `time`, or None for a miss.

        Its own prefix, because a thumbnail and a canvas frame can be the same
        moment of the same video at two very different sizes (72 rows against
        720): one name for both would hand the queue a canvas frame, or the
        canvas a 72-row one."""
        key = _time_key(time)
        return None if key is None else self._read(_thumb_name(key))

    def write_thumbnail(self, time: float, image: np.ndarray) -> bool:
        """Store `image` as the queue thumbnail at `time`, losslessly.

        Lossless costs nothing worth counting here -- a 128x72 thumbnail is a
        few KB either way, against 84 KB for a canvas frame -- and it keeps
        the one picture the user sees hundreds of at a time free of artefacts
        at the size where they would show most."""
        key = _time_key(time)
        if key is None:
            return False
        return self._write(_thumb_name(key), image, [cv2.IMWRITE_WEBP_QUALITY, LOSSLESS_QUALITY])

    def read_strip(self, crop_box: Box, time: float) -> np.ndarray | None:
        """The cached OCR-exact strip at `time` for `crop_box`, or None."""
        box_key, key = _box_key(crop_box), _time_key(time)
        if box_key is None or key is None:
            return None
        return self._read(_strip_name(box_key, key))

    def write_strip(self, crop_box: Box, time: float, image: np.ndarray) -> bool:
        """Store `image` as the OCR-exact strip at `time` for `crop_box`,
        losslessly; True when it landed."""
        box_key, key = _box_key(crop_box), _time_key(time)
        if box_key is None or key is None:
            return False
        return self._write(_strip_name(box_key, key), image,
                           [int(cv2.IMWRITE_WEBP_QUALITY), LOSSLESS_QUALITY])

    def _read(self, name: str) -> np.ndarray | None:
        if not self._readable:
            return None
        try:
            data = (self.directory / name).read_bytes()
        except OSError:
            return None                                          # a miss and an unreadable file read the same
        if not data:
            return None
        try:
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:                                 # pragma: no cover - imdecode returns None
            logger.debug("View cache could not decode %s (%s)", name, exc)
            return None

    def _write(self, name: str, image: np.ndarray, params: list[int]) -> bool:
        if not self._readable:
            return False
        data = _encode(image, params)
        if data is None or not self._ensure_meta():
            return False
        return _atomic_write(self.directory / name, data)

    # -- housekeeping -----------------------------------------------------

    def trim(self, keep: Wanted) -> int:
        """Delete every cached image `keep` does not name -- times that are no
        longer in evidence, and strips of an old crop box -- and return how
        many went. `meta.json` is never trimmed."""
        if not self._readable:
            return 0
        frame_times = {float(key) for key in map(_time_key, keep.frame_times) if key is not None}
        thumb_key = None if keep.thumbnail_time is None else _time_key(keep.thumbnail_time)
        thumb_time = None if thumb_key is None else float(thumb_key)
        box_key = _box_key(keep.crop_box)
        strip_times = ({float(key) for key in map(_time_key, keep.strip_times) if key is not None}
                       if box_key is not None else set())
        try:
            names = [entry.name for entry in os.scandir(self.directory)]
        except OSError:
            return 0
        deleted = 0
        for name in names:
            if not _IMAGE_NAME.match(name) or self._keeps(name, frame_times, box_key, strip_times,
                                                          thumb_time):
                continue
            try:
                os.unlink(self.directory / name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.debug("View cache could not trim %s (%s)", name, exc)
                continue
            deleted += 1
        return deleted

    @staticmethod
    def _keeps(name: str, frame_times: set[float], box_key: str | None, strip_times: set[float],
               thumb_time: float | None = None) -> bool:
        """Whether `name` is one of the images still wanted. A name of the
        cache's own shape that will not parse is not one of them: nothing can
        ever ask for it, so it is only taking up room."""
        thumb = _THUMB_NAME.match(name)
        if thumb is not None:
            return thumb_time is not None and float(thumb.group(1)) == thumb_time
        frame = _FRAME_NAME.match(name)
        if frame is not None:
            return float(frame.group(1)) in frame_times
        strip = _STRIP_NAME.match(name)
        if strip is not None:
            return strip.group(1) == box_key and float(strip.group(2)) in strip_times
        return False
