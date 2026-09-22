"""The on-disk store for the review views' pixels (core.jobs.view_cache).

Two guarantees are worth more than the rest of this file:

- a strip round-trips bit-exact. The Brightness tab measures a threshold on
  those pixels, so a lossy round trip would move the number the user is
  looking at. WebP lossless (quality 101) is the reason it holds, and the
  tests below pin it on noise and on a subtitle-like high-contrast image;
- nothing the cache does can fail a job. A read-only project, a full disk, a
  half-written file, a video that has been replaced: every one of them has to
  degrade to "decode it again", never to an exception and never to the wrong
  picture.

The rest pins the layout (`f-<time>.webp`, `s-<box>-<time>.webp` under
`.ocr-cache/view/<name digest>/`), the meta.json identity check and what
`trim` keeps.

No video is decoded here: the cache only stat()s its source, so an ordinary
file standing in for one is enough.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.jobs.view_cache import (
    FORMAT_VERSION,
    FRAME_QUALITY,
    FileViewCache,
    Wanted,
    prune,
    view_dir,
    wanted,
)
from core.project.model import Crop, FileEntry, Source
from core.project.store import evidence_path

BOX = (288, 786, 1344, 53)
OTHER_BOX = (0, 0, 640, 48)
NAME = "Episode 01.mkv"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _video(directory: Path, name: str = NAME, size: int = 2048) -> Path:
    """A stand-in for the episode: the cache only ever stat()s it."""
    path = directory / name
    path.write_bytes(b"\0" * size)
    return path


def _cache(directory: Path, name: str = NAME) -> FileViewCache:
    return FileViewCache(str(directory), name, str(directory / name))


def _noise(height: int = 53, width: int = 1344) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _subtitle_strip(height: int = 53, width: int = 1344) -> np.ndarray:
    """A strip that looks like what the OCR pass reads: near-black background,
    hard white glyph blocks with a grey outline -- the levels the brightness
    threshold is chosen between."""
    image = np.full((height, width, 3), 8, dtype=np.uint8)
    for start in range(120, width - 120, 90):
        image[8:45, start - 3:start + 51] = 90                  # outline
        image[11:42, start:start + 48] = 255                    # glyph
        image[20:33, start + 12:start + 36] = 8                 # a hole in it
    return image


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _images(directory: Path) -> list[str]:
    return sorted(name for name in _names(directory) if name.endswith(".webp"))


# --------------------------------------------------------------------------
# Where it lives
# --------------------------------------------------------------------------

def test_the_view_directory_is_the_name_digest_under_the_project_cache(tmp_path):
    digest = hashlib.sha256(os.fsencode(NAME)).hexdigest()

    assert view_dir(str(tmp_path), NAME) == tmp_path / ".ocr-cache" / "view" / digest


def test_the_view_directory_uses_the_same_digest_as_the_evidence_cache(tmp_path):
    """One convention for both caches: whatever name a file has (spaces,
    Chinese, a byte that is not UTF-8), it hashes to the same short directory
    name the evidence file is called."""
    for name in (NAME, "第01集.mkv", "a/b is not a name.mp4"):
        assert view_dir(str(tmp_path), name).name == evidence_path(str(tmp_path), name).stem


def test_the_view_directory_is_not_created_until_something_is_written(tmp_path):
    _video(tmp_path)

    cache = _cache(tmp_path)

    assert cache.readable is True
    assert not view_dir(str(tmp_path), NAME).exists()


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------

def test_a_frame_comes_back_the_same_shape_but_not_the_same_pixels(tmp_path):
    """Frames are a canvas to look at, so they are stored lossily (~84 KB
    against 571 KB): the same shape and dtype come back, not the same
    values."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    frame = _noise(180, 320)                                     # noise: the worst case WebP has

    assert cache.write_frame(120.0, frame) is True
    back = cache.read_frame(120.0)

    assert back is not None
    assert back.shape == frame.shape and back.dtype == frame.dtype
    assert not np.array_equal(back, frame)                       # lossy, by design


def test_a_frame_of_a_real_looking_picture_survives_the_lossy_round_trip(tmp_path):
    """And close enough to look at: a picture with structure comes back
    visibly the same."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    frame = _subtitle_strip(180, 320)

    cache.write_frame(1.5, frame)
    back = cache.read_frame(1.5)

    assert back.shape == frame.shape and back.dtype == frame.dtype
    assert float(np.abs(back.astype(int) - frame.astype(int)).mean()) < 8


def test_a_strip_round_trips_bit_exact(tmp_path):
    """The fidelity guarantee: the Brightness tab measures its threshold on
    these pixels, so they are stored losslessly and must come back identical."""
    _video(tmp_path)
    cache = _cache(tmp_path)

    for strip in (_noise(), _subtitle_strip()):
        assert cache.write_strip(BOX, 120.0, strip) is True
        back = cache.read_strip(BOX, 120.0)

        assert back is not None
        assert back.dtype == strip.dtype and back.shape == strip.shape
        assert np.array_equal(back, strip)


def test_a_strip_is_only_read_back_for_the_box_it_was_written_with(tmp_path):
    """Strips are valid for one crop box only: another box is a miss, not the
    old pixels."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_strip(BOX, 120.0, _subtitle_strip())

    assert cache.read_strip(OTHER_BOX, 120.0) is None
    assert cache.read_strip(BOX, 120.0) is not None


def test_a_frame_and_a_strip_at_the_same_time_do_not_collide(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    frame, strip = _noise(180, 320), _subtitle_strip()

    cache.write_frame(120.0, frame)
    cache.write_strip(BOX, 120.0, strip)

    assert np.array_equal(cache.read_strip(BOX, 120.0), strip)
    assert cache.read_frame(120.0).shape == frame.shape


def test_writing_the_same_time_again_replaces_what_was_there(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    first, second = _subtitle_strip(), _noise()

    cache.write_strip(BOX, 9.0, first)
    assert cache.write_strip(BOX, 9.0, second) is True

    assert np.array_equal(cache.read_strip(BOX, 9.0), second)
    assert len(_images(view_dir(str(tmp_path), NAME))) == 1


def test_an_empty_cache_misses(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)

    assert cache.read_frame(120.0) is None
    assert cache.read_strip(BOX, 120.0) is None


def test_a_time_that_was_never_written_misses(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(120.0, _noise(180, 320))

    assert cache.read_frame(120.001) is None


# --------------------------------------------------------------------------
# File names: the time and the box
# --------------------------------------------------------------------------

def test_the_file_names_spell_out_the_time_and_the_crop_box(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)

    cache.write_frame(120.0, _noise(64, 64))
    cache.write_strip(BOX, 120.0, _subtitle_strip())

    assert _images(view_dir(str(tmp_path), NAME)) == ["f-000120.000.webp",
                                                      "s-288_786_1344_53-000120.000.webp"]


def test_the_names_sort_by_time(tmp_path):
    """Fixed width, so a listing of the directory reads in time order."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    times = [9.5, 120.0, 1000.25]

    for time in times:
        cache.write_frame(time, _noise(32, 32))

    names = _images(view_dir(str(tmp_path), NAME))
    assert names == sorted(names)
    assert [float(name[2:-5]) for name in names] == times


def test_two_times_a_millisecond_apart_are_two_files(tmp_path):
    """Times are ms-resolution in practice, and the name keeps every one of
    them apart: neighbouring times must never share a file."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    first, second = _subtitle_strip(), _noise()

    cache.write_strip(BOX, 120.001, first)
    cache.write_strip(BOX, 120.002, second)

    assert len(_images(view_dir(str(tmp_path), NAME))) == 2
    assert np.array_equal(cache.read_strip(BOX, 120.001), first)
    assert np.array_equal(cache.read_strip(BOX, 120.002), second)


def test_a_time_the_name_cannot_hold_exactly_is_not_cached(tmp_path):
    """Rather than round 1/3 s onto its neighbour's file and hand back the
    wrong frame, a time the name cannot spell is simply not cached: the view
    decodes it, as it did before the cache existed."""
    _video(tmp_path)
    cache = _cache(tmp_path)

    assert cache.write_frame(1 / 3, _noise(32, 32)) is False
    assert cache.write_strip(BOX, 1 / 3, _subtitle_strip()) is False
    assert cache.read_frame(1 / 3) is None
    assert cache.read_strip(BOX, 1 / 3) is None
    assert not view_dir(str(tmp_path), NAME).exists() or _images(view_dir(str(tmp_path), NAME)) == []


def test_a_time_that_is_not_a_number_is_not_cached(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)

    assert cache.write_frame(float("nan"), _noise(32, 32)) is False
    assert cache.write_frame(float("inf"), _noise(32, 32)) is False
    assert cache.read_frame(float("nan")) is None


# --------------------------------------------------------------------------
# meta.json: the video the pixels came from
# --------------------------------------------------------------------------

def _meta(tmp_path: Path) -> dict:
    return json.loads((view_dir(str(tmp_path), NAME) / "meta.json").read_text())


def test_the_cache_records_the_format_version_and_the_source(tmp_path):
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))

    stat = video.stat()
    assert _meta(tmp_path) == {"version": FORMAT_VERSION, "size": stat.st_size,
                               "mtime_ns": stat.st_mtime_ns}


def test_a_video_of_another_size_wipes_the_pixels(tmp_path):
    """A cached frame of a video that has been replaced would show the wrong
    picture, silently: the directory is emptied instead."""
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    cache.write_strip(BOX, 1.0, _subtitle_strip())
    video.write_bytes(b"\0" * 4096)

    fresh = _cache(tmp_path)

    assert fresh.read_frame(1.0) is None
    assert fresh.read_strip(BOX, 1.0) is None
    assert _images(view_dir(str(tmp_path), NAME)) == []
    assert _meta(tmp_path)["size"] == 4096


def test_a_video_of_the_same_size_but_another_mtime_wipes_the_pixels(tmp_path):
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    stat = video.stat()
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    fresh = _cache(tmp_path)

    assert fresh.read_frame(1.0) is None
    assert _images(view_dir(str(tmp_path), NAME)) == []


def test_an_unchanged_video_keeps_its_pixels(tmp_path):
    _video(tmp_path)
    strip = _subtitle_strip()
    _cache(tmp_path).write_strip(BOX, 1.0, strip)

    assert np.array_equal(_cache(tmp_path).read_strip(BOX, 1.0), strip)


def test_a_missing_or_unreadable_meta_wipes_the_pixels(tmp_path):
    """No identity, no trust: a meta.json that is gone, junk or from another
    format version says nothing about which video these pixels came from."""
    for damage in (lambda path: path.unlink(),
                   lambda path: path.write_text("{ not json"),
                   lambda path: path.write_text(json.dumps({"version": FORMAT_VERSION + 1,
                                                            "size": 2048, "mtime_ns": 0})),
                   lambda path: path.write_text(json.dumps(["not", "an", "object"]))):
        _video(tmp_path)
        _cache(tmp_path).write_frame(1.0, _noise(32, 32))
        damage(view_dir(str(tmp_path), NAME) / "meta.json")

        assert _cache(tmp_path).read_frame(1.0) is None
        assert _images(view_dir(str(tmp_path), NAME)) == []


def test_a_video_that_cannot_be_stat_d_is_not_readable(tmp_path):
    """Without the video there is nothing to check the pixels against, so the
    cache switches itself off rather than guess."""
    cache = _cache(tmp_path)                                     # no video file at all

    assert cache.readable is False
    assert cache.read_frame(1.0) is None
    assert cache.read_strip(BOX, 1.0) is None
    assert cache.write_frame(1.0, _noise(32, 32)) is False
    assert cache.write_strip(BOX, 1.0, _subtitle_strip()) is False
    assert cache.trim(Wanted((1.0,), BOX, (1.0,))) == 0
    assert not view_dir(str(tmp_path), NAME).exists()


# --------------------------------------------------------------------------
# Failing I/O degrades, it never raises
# --------------------------------------------------------------------------

@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_project_that_cannot_be_written_to_degrades_to_a_miss(tmp_path):
    """A read-only mount or a full disk must cost nothing but the decode it
    was costing before the cache existed."""
    project = tmp_path / "read-only"
    project.mkdir()
    _video(project)
    project.chmod(0o500)
    try:
        cache = _cache(project)

        assert cache.readable is True                            # the video itself is fine
        assert cache.write_frame(1.0, _noise(32, 32)) is False
        assert cache.write_strip(BOX, 1.0, _subtitle_strip()) is False
        assert cache.read_frame(1.0) is None
        assert cache.trim(Wanted((), None, ())) == 0
    finally:
        project.chmod(0o700)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_cache_directory_that_cannot_be_written_to_degrades_to_a_miss(tmp_path):
    """The project is writable but its view directory is not: same answer."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    directory = view_dir(str(tmp_path), NAME)
    directory.chmod(0o500)
    try:
        assert cache.write_frame(2.0, _noise(32, 32)) is False
        assert cache.read_frame(1.0) is not None                 # reading still works
    finally:
        directory.chmod(0o700)


def test_a_corrupt_image_reads_as_a_miss(tmp_path):
    """A truncated or junk file is a miss, not an exception: whatever left it
    there, the view just decodes the time again."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_strip(BOX, 1.0, _subtitle_strip())
    cache.write_frame(1.0, _noise(32, 32))
    directory = view_dir(str(tmp_path), NAME)
    strip_file = directory / "s-288_786_1344_53-000001.000.webp"
    strip_file.write_bytes(strip_file.read_bytes()[: len(strip_file.read_bytes()) // 2])
    (directory / "f-000001.000.webp").write_bytes(b"not a webp at all")

    assert cache.read_strip(BOX, 1.0) is None
    assert cache.read_frame(1.0) is None


def test_an_empty_image_file_reads_as_a_miss(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    (view_dir(str(tmp_path), NAME) / "f-000001.000.webp").write_bytes(b"")

    assert cache.read_frame(1.0) is None


def test_a_write_that_fails_at_the_last_step_leaves_nothing_behind(tmp_path, monkeypatch):
    """Writes land with os.replace, so a reader either sees the whole image or
    no image: a half-written WebP would draw a torn frame. When the replace
    fails, the temporary file goes with it."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))                       # the directory and meta exist

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)

    assert cache.write_frame(2.0, _noise(32, 32)) is False
    assert cache.write_strip(BOX, 2.0, _subtitle_strip()) is False
    assert _names(view_dir(str(tmp_path), NAME)) == ["f-000001.000.webp", "meta.json"]


def test_an_image_the_encoder_refuses_is_not_a_crash(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)

    assert cache.write_frame(1.0, np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert cache.write_strip(BOX, 1.0, np.zeros((4, 4, 5), dtype=np.uint8)) is False
    assert cache.write_frame(1.0, None) is False
    assert cache.read_frame(1.0) is None


# --------------------------------------------------------------------------
# trim
# --------------------------------------------------------------------------

def test_trim_keeps_what_is_wanted_and_deletes_the_rest(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    for time in (1.0, 2.0, 3.0):
        cache.write_frame(time, _noise(32, 32))
    for time in (1.0, 2.0):
        cache.write_strip(BOX, time, _subtitle_strip())

    deleted = cache.trim(Wanted(frame_times=(2.0,), crop_box=BOX, strip_times=(1.0,)))

    assert deleted == 3                                          # frames 1.0 and 3.0, strip 2.0
    assert cache.read_frame(2.0) is not None
    assert cache.read_strip(BOX, 1.0) is not None
    assert cache.read_frame(1.0) is None and cache.read_frame(3.0) is None
    assert cache.read_strip(BOX, 2.0) is None


def test_trim_never_deletes_the_meta(tmp_path):
    """meta.json is the identity of the pixels, not one of them."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))

    cache.trim(Wanted((), None, ()))

    assert _names(view_dir(str(tmp_path), NAME)) == ["meta.json"]


def test_trim_deletes_the_strips_of_an_old_crop_box(tmp_path):
    """The crop moved: every strip on disk shows the wrong pixels for the box
    the Brightness tab now measures on."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_strip(BOX, 1.0, _subtitle_strip())
    cache.write_strip(BOX, 2.0, _subtitle_strip())
    cache.write_frame(1.0, _noise(32, 32))

    deleted = cache.trim(Wanted(frame_times=(1.0,), crop_box=OTHER_BOX, strip_times=(1.0, 2.0)))

    assert deleted == 2
    assert _images(view_dir(str(tmp_path), NAME)) == ["f-000001.000.webp"]


def test_trim_with_no_crop_box_deletes_every_strip(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_strip(BOX, 1.0, _subtitle_strip())
    cache.write_frame(1.0, _noise(32, 32))

    assert cache.trim(Wanted(frame_times=(1.0,), crop_box=None, strip_times=(1.0,))) == 1
    assert _images(view_dir(str(tmp_path), NAME)) == ["f-000001.000.webp"]


def test_trim_keeps_everything_it_is_asked_to(tmp_path):
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    cache.write_strip(BOX, 1.0, _subtitle_strip())

    assert cache.trim(Wanted((1.0,), BOX, (1.0,))) == 0
    assert len(_images(view_dir(str(tmp_path), NAME))) == 2


def test_trim_on_a_cache_with_nothing_in_it_deletes_nothing(tmp_path):
    _video(tmp_path)

    assert _cache(tmp_path).trim(Wanted((1.0,), BOX, (1.0,))) == 0


def test_trim_leaves_files_that_are_not_its_own_alone(tmp_path):
    """Something else's file in the directory is not the cache's to delete;
    one of its own that it cannot read back is."""
    _video(tmp_path)
    cache = _cache(tmp_path)
    cache.write_frame(1.0, _noise(32, 32))
    directory = view_dir(str(tmp_path), NAME)
    (directory / "notes.txt").write_text("hello")
    (directory / "f-nonsense.webp").write_bytes(b"junk")

    deleted = cache.trim(Wanted((1.0,), None, ()))

    assert deleted == 1
    assert _names(directory) == ["f-000001.000.webp", "meta.json", "notes.txt"]


# --------------------------------------------------------------------------
# prune: the directories of videos that have left the folder
# --------------------------------------------------------------------------

def _warm(directory: Path, name: str) -> Path:
    """A file's view directory, with something in it."""
    _video(directory, name)
    cache = FileViewCache(str(directory), name, str(directory / name))
    cache.write_frame(1.0, _noise(32, 32))
    return view_dir(str(directory), name)


def test_prune_deletes_the_directory_of_a_video_that_has_left_the_folder(tmp_path):
    """Nothing else ever removes these: a deleted or renamed episode would
    keep its pixels for good."""
    gone = _warm(tmp_path, "Episode 01.mkv")
    stayed = _warm(tmp_path, "Episode 02.mkv")

    assert prune(str(tmp_path), ["Episode 02.mkv"]) == 1

    assert not gone.exists()
    assert stayed.exists()


def test_prune_keeps_every_video_still_in_the_folder(tmp_path):
    kept = [_warm(tmp_path, name) for name in ("Episode 01.mkv", "Episode 02.mkv")]

    assert prune(str(tmp_path), ["Episode 01.mkv", "Episode 02.mkv"]) == 0
    assert all(directory.exists() for directory in kept)


def test_prune_of_an_empty_folder_takes_everything(tmp_path):
    first, second = _warm(tmp_path, "Episode 01.mkv"), _warm(tmp_path, "Episode 02.mkv")

    assert prune(str(tmp_path), []) == 2
    assert not first.exists() and not second.exists()


def test_prune_leaves_anything_that_is_not_one_of_its_directories(tmp_path):
    """A digest-shaped directory is the cache's; a loose file or a directory
    under any other name belongs to somebody else."""
    _warm(tmp_path, NAME)
    view = view_dir(str(tmp_path), NAME).parent
    (view / "notes.txt").write_text("hello")
    (view / "not-a-digest").mkdir()
    (view / ("a" * 64)).write_text("a file, not a directory")

    assert prune(str(tmp_path), []) == 1
    assert sorted(path.name for path in view.iterdir()) == ["a" * 64, "not-a-digest", "notes.txt"]


def test_prune_without_a_view_directory_does_nothing(tmp_path):
    """A project nobody has opened a tab on yet."""
    assert prune(str(tmp_path), []) == 0
    assert prune(str(tmp_path), [NAME]) == 0
    assert not (tmp_path / ".ocr-cache" / "view").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_prune_that_cannot_delete_reports_what_it_managed(tmp_path):
    """A read-only cache is one more thing that costs a decode, not a
    failure."""
    first = _warm(tmp_path, "Episode 01.mkv")
    view = first.parent
    view.chmod(0o500)
    try:
        assert prune(str(tmp_path), []) == 0
        assert first.exists()
    finally:
        view.chmod(0o700)


# --------------------------------------------------------------------------
# wanted(): the times a file's tabs draw
# --------------------------------------------------------------------------

def _entry(evidence: dict, crop: Crop | None = None) -> FileEntry:
    return FileEntry(name=NAME, crop=crop, evidence=evidence)


def test_wanted_reads_the_crop_evidence_for_the_frames(tmp_path):
    entry = _entry({"crop": {"samples": [{"time": 2.0}, {"time": 1.0}, {"time": 2.0}]}})

    assert wanted(entry).frame_times == (2.0, 1.0)               # in order, de-duplicated


def test_wanted_reads_the_tiles_and_the_gallery_for_the_strips(tmp_path):
    entry = _entry({"brightness": {"tiles": {"dark": 3.0, "bright": 4.0}},
                    "lines": {"samples": [{"time": 5.0}, {"time": 3.0}]}},
                   crop=Crop(*BOX, source=Source.DETECTED))

    keep = wanted(entry)

    assert keep.crop_box == BOX
    assert keep.strip_times == (3.0, 4.0, 5.0)


def test_wanted_drops_evidence_that_is_missing_or_malformed(tmp_path):
    """Evidence is a disposable cache: a junk entry contributes no times
    instead of raising on a view repaint."""
    entry = _entry({"crop": {"samples": [{"time": "no"}, "junk", {}, {"time": 1.0}]},
                    "brightness": {"tiles": ["not", "a", "mapping"]},
                    "lines": None})

    assert wanted(entry) == Wanted(frame_times=(1.0,), crop_box=None, strip_times=())


def test_a_file_with_nothing_to_hold_is_falsy():
    assert not Wanted((), None, ())
    assert not Wanted((), None, (1.0,))                          # strips need a box
    assert not Wanted((), BOX, ())
    assert Wanted((1.0,), None, ())
    assert Wanted((), BOX, (1.0,))


# --------------------------------------------------------------------------
# The core boundary
# --------------------------------------------------------------------------

def test_view_cache_imports_no_qt():
    code = ("import sys, core.jobs.view_cache; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_frames_are_stored_lossily_and_strips_are_not(tmp_path):
    """The size argument for the two settings, on one canvas-sized picture:
    the frame is worth a fraction of the lossless file (~84 KB against
    ~571 KB on a real episode), and the strip the threshold is measured on is
    worth being exact."""
    assert FRAME_QUALITY < 100
    _video(tmp_path)
    cache = _cache(tmp_path)
    picture = _noise(720, 1280)

    cache.write_frame(1.0, picture)
    cache.write_strip(BOX, 1.0, picture)

    directory = view_dir(str(tmp_path), NAME)
    lossy = (directory / "f-000001.000.webp").stat().st_size
    lossless = (directory / "s-288_786_1344_53-000001.000.webp").stat().st_size
    assert lossy * 2 < lossless
    assert np.array_equal(cache.read_strip(BOX, 1.0), picture)
