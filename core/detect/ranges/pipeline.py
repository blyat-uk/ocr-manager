"""Folder-level time-range detection: ingest, segment discovery, keep ranges.

Same algorithm and bit-identical output as the original
``core/audio_analysis.py`` + ``core/audio_finder/``, minus the waste:

* file identity is size + mtime + a hash of the first and last 4 MB instead
  of a full-file SHA256 (which was 11-40 % of a run, purely as a cache key);
* fingerprinting runs in a process pool (ffmpeg decode was 45-73 %, serial);
* fingerprints are cached as one ``.npz`` per file instead of SQLite rows;
* discovery and propagation use the vectorised joins in ``matching``.

Output equivalence is defined against the original running on a fresh
database. Two deliberate differences from the original's *persistent*
database: files are analysed exactly as passed in (the old database kept
every file it had ever ingested for the folder, including renamed or
deleted ones), and two byte-identical files are both analysed (the old
SHA256 key silently dropped the second).
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np

from core.detect.ranges import fingerprint
from core.detect.ranges import matching as mt
from core.detect.ranges.config import RangesConfig

logger = logging.getLogger(__name__)

# Minimum repeating-segment length the app uses (seconds).
DEFAULT_MIN_SEGMENT_SEC = 30.0
# Minimum gap duration (seconds) to be considered a "keep" range.
MIN_GAP_SEC = 5.0
# Fingerprinting processes: 8 at most, never more than the machine has CPUs.
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
CACHE_DIRNAME = ".ocr-cache"

_MIN_FILES_FOR_STOP_WORDS = 10
_MAX_ITERATIONS = 50
_IDENTITY_EDGE_BYTES = 4 * 1024 * 1024
_CACHE_SUBDIR = "fingerprints"
_CACHE_FORMAT = 1


@dataclass(frozen=True)
class FileEntry:
    name: str   # key in the result dict (the project-relative filename)
    path: str   # absolute path to read


@dataclass(frozen=True)
class ProgressEvent:
    """kind is "phase" (message = phase name), "file" (message = filename,
    current/total = files fingerprinted so far), or "log" (a discovery log
    line, same text the original emitted)."""
    kind: str
    message: str
    current: int = 0
    total: int = 0


class AnalysisCancelled(Exception):
    """Raised by analyse() when its cancel callback returns True."""


@dataclass(frozen=True)
class SegmentMatch:
    file: int           # index into the analysed files
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    score: float


@dataclass(frozen=True)
class Segment:
    pivot: int          # index of the file the canonical segment came from
    candidate: int      # index of the file it was discovered against
    start_frame: int
    end_frame: int
    duration_sec: float
    matches: tuple[SegmentMatch, ...]


@dataclass(frozen=True)
class Block:
    """One repeating-segment match, as evidence for the review UI.

    Blocks are the RAW matched segments per file -- possibly overlapping,
    never merged. This is deliberately different from compute_keep_ranges(),
    which inverts each file's *merged* skip blocks into keep spans; a Block
    here is one SegmentMatch, one-to-one, so the UI can show exactly what
    was matched (and against how many other files, and how well) rather
    than the collapsed result. See compute_blocks().
    """
    start_sec: float
    end_sec: float
    kind: str            # "intro" | "outro" | "repeat"
    matched_files: int    # distinct files this segment was matched in, including this one
    score: float          # this file's own SegmentMatch.score


@dataclass(frozen=True)
class RangesAnalysis:
    keep: dict[str, list[tuple[str | None, str | None]]]   # identical to analyse()'s return
    blocks: dict[str, list[Block]]                          # per filename, sorted by start_sec
    durations: dict[str, float]


ProgressFn = Callable[[ProgressEvent], None]
CancelFn = Callable[[], bool]


def _check_cancel(cancel: CancelFn | None) -> None:
    if cancel is not None and cancel():
        raise AnalysisCancelled()


# ---------------------------------------------------------------------------
# Identity and cache
# ---------------------------------------------------------------------------

def file_identity(path: str) -> str:
    """Cheap content identity: size, mtime and SHA256 of the first and last
    4 MB. A change confined to the middle of a file with size and mtime
    preserved is not seen -- an accepted trade for reading 8 MB instead of
    the whole file."""
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{st.st_mtime_ns}:".encode())
    with open(path, "rb") as f:
        h.update(f.read(_IDENTITY_EDGE_BYTES))
        if st.st_size > _IDENTITY_EDGE_BYTES:
            f.seek(max(_IDENTITY_EDGE_BYTES, st.st_size - _IDENTITY_EDGE_BYTES))
            h.update(f.read(_IDENTITY_EDGE_BYTES))
    return h.hexdigest()


def cache_key(identity: str, cfg: RangesConfig) -> str:
    payload = json.dumps(
        {"format": _CACHE_FORMAT, "identity": identity, "params": cfg.fingerprint_params()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def default_cache_dir(project_dir: str) -> str:
    return os.path.join(project_dir, CACHE_DIRNAME)


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, _CACHE_SUBDIR, f"{key}.npz")


def load_cached(cache_dir: str, key: str) -> tuple[np.ndarray, float] | None:
    """Cached (hashes, duration), or None on a miss or an unreadable entry."""
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            hashes = data["hashes"]
            duration = data["duration"]
    except Exception:  # noqa: BLE001 - any unreadable entry is just a miss
        logger.warning("Ignoring unreadable fingerprint cache entry %s", path)
        return None
    if hashes.dtype != np.int64 or hashes.ndim != 2 or hashes.shape[1] != 2 \
            or duration.shape != () or duration.dtype != np.float64:
        return None
    return hashes, float(duration)


def save_cached(cache_dir: str, key: str, hashes: np.ndarray, duration: float) -> None:
    """Atomic write; a failure is logged and otherwise ignored."""
    path = _cache_path(cache_dir, key)
    directory = os.path.dirname(path)
    tmp = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            np.savez(f, hashes=hashes, duration=np.float64(duration))
        os.replace(tmp, path)
        tmp = None
    except OSError:
        logger.warning("Could not write fingerprint cache entry %s", path, exc_info=True)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _pool_context():
    # forkserver: never fork() the (possibly multi-threaded, Qt) caller.
    # Windows has no forkserver (and macOS's fork is unsafe for the same
    # reason): spawn there. A spawned worker imports the fingerprint module
    # afresh to unpickle fingerprint_file, so it needs no preload -- the
    # preload below is a forkserver-only setting.
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context("forkserver")
    # "__main__" is preloaded alongside the fingerprint module so the
    # forkserver process (spawned once, reused for every task) pays the
    # cost of importing the app's entry module a single time instead of on
    # every worker fork. This is safe: the forkserver bootstrap reimports
    # "__main__" via multiprocessing.spawn's path-based fixup, which runs
    # the module under run_name="__mp_main__" -- every entry point this
    # preload can see (main.py, `python -m pytest`, the `pytest` console
    # script, tools/bench.py) guards its side effects behind
    # `if __name__ == "__main__":`, so that guard is False and nothing is
    # re-executed. Verified directly: a forkserver pool with "__main__" in
    # the preload list, started both under pytest and as a standalone
    # script, completes without hanging or re-running the script's own
    # `if __name__ == "__main__":` block (see task-6-report.md).
    ctx.set_forkserver_preload(["core.detect.ranges.fingerprint", "__main__"])
    return ctx


def ingest(
    files: Sequence[FileEntry],
    cfg: RangesConfig,
    *,
    cache_dir: str | None = None,
    workers: int = DEFAULT_WORKERS,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Fingerprint every file (cache first, then a process pool)."""
    total = len(files)
    hashes: list[np.ndarray | None] = [None] * total
    durations: list[float] = [0.0] * total
    keys: list[str | None] = [None] * total
    done = 0

    def finish(i: int, result: tuple[np.ndarray, float], from_cache: bool) -> None:
        nonlocal done
        hashes[i], durations[i] = result
        if cache_dir is not None and not from_cache:
            save_cached(cache_dir, keys[i], *result)
        done += 1
        if progress is not None:
            progress(ProgressEvent("file", files[i].name, done, total))

    misses = []
    for i, entry in enumerate(files):
        _check_cancel(cancel)
        if cache_dir is not None:
            keys[i] = cache_key(file_identity(entry.path), cfg)
            cached = load_cached(cache_dir, keys[i])
            if cached is not None:
                finish(i, cached, from_cache=True)
                continue
        misses.append(i)

    if workers <= 1 or len(misses) <= 1:
        for i in misses:
            _check_cancel(cancel)
            finish(i, fingerprint.fingerprint_file(files[i].path, cfg), from_cache=False)
    elif misses:
        # Two separate try/except boundaries, deliberately: a failure
        # creating/starting the pool (or submitting to it) means "the pool
        # never ran anything" -- fully safe to retry every miss serially.
        # A failure surfacing later, while collecting results, could mean
        # one of two very different things, and they must not be conflated:
        # a BrokenProcessPool (a worker process died, e.g. OOM-killed) is
        # still an infrastructure failure -- fall back for whatever is not
        # yet finished. Any OTHER exception from future.result() is the
        # fingerprint function's own exception for that one file (a real
        # decode/IO error, possibly an OSError subclass like
        # FileNotFoundError) -- that must propagate exactly as it did
        # before this fallback existed, with no fallback warning and no
        # retry, or a real per-file error would be silently retried and
        # misreported as "the pool is unavailable."
        pool = None
        try:
            pool = ProcessPoolExecutor(max_workers=min(workers, len(misses)), mp_context=_pool_context())
            pending = {pool.submit(fingerprint.fingerprint_file, files[i].path, cfg): i for i in misses}
        except (ValueError, OSError, BrokenProcessPool) as exc:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            pool = None
            logger.warning(
                "Process pool could not start (%s: %s); falling back to serial fingerprinting.",
                type(exc).__name__, exc,
            )
        except BaseException:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            raise

        if pool is not None:
            try:
                while pending:
                    _check_cancel(cancel)
                    completed, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        finish(pending.pop(future), future.result(), from_cache=False)
            except BrokenProcessPool as exc:
                pool.shutdown(wait=False, cancel_futures=True)
                finished_count = sum(1 for i in misses if hashes[i] is not None)
                logger.warning(
                    "Process pool crashed after %d of %d files (%s); falling back to serial "
                    "fingerprinting for the rest.",
                    finished_count, len(misses), exc,
                )
            except BaseException:
                # Anything else here is the fingerprint function's OWN
                # exception for one file (surfaced via future.result()),
                # not a pool infrastructure failure -- shut down and
                # propagate unchanged, exactly as before this fallback
                # existed. No fallback warning, no retry.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)

        # Reached when the pool never started, or broke mid-run: fingerprint
        # whatever is still missing serially, in-process, with the exact
        # same function -- identical fingerprints, just no parallelism. A
        # no-op when every miss already finished through the pool.
        for i in misses:
            if hashes[i] is None:
                _check_cancel(cancel)
                finish(i, fingerprint.fingerprint_file(files[i].path, cfg), from_cache=False)

    return hashes, durations  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

def mask_range(mask: np.ndarray, start: int, end: int) -> None:
    """The original MaskSet.mask_range, including its Python slice semantics
    (a negative ``end + 1`` counts from the back of the mask)."""
    s = max(0, start)
    e = min(len(mask), end + 1)
    mask[s:e] = True


def unmasked_hashes(
    hashes: np.ndarray, times: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprints whose time is inside the mask and not yet masked."""
    keep = times < len(mask)
    keep[keep] = ~mask[times[keep]]
    return hashes[keep], times[keep]


def discover_segments(
    fingerprints: Sequence[np.ndarray],
    durations: Sequence[float],
    cfg: RangesConfig,
    log: Callable[[str], None] | None = None,
    cancel: CancelFn | None = None,
) -> list[Segment]:
    """The original iterative pivot/propagate loop.

    Log lines are the original's text; file numbers in them are 1-based
    positions in ``fingerprints`` (what the original's database ids were on
    a fresh database).
    """
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    n = len(fingerprints)
    if n < 2:
        return []

    dsp, hcfg, m = cfg.dsp, cfg.hash, cfg.match
    fps = dsp.frames_per_sec
    arrays = [np.asarray(f, dtype=np.int64).reshape(-1, 2) for f in fingerprints]

    if n >= _MIN_FILES_FOR_STOP_WORDS:
        stop = mt.stop_word_hashes(arrays, m.stop_word_file_ratio)
    else:
        stop = np.empty(0, dtype=np.int64)
    _log(f"  Stop-word hashes: {len(stop):,}")

    file_h, file_t = [], []
    for a in arrays:
        h, t = a[:, 0], a[:, 1]
        if stop.size:
            keep = ~np.isin(h, stop)
            h, t = h[keep], t[keep]
        file_h.append(np.ascontiguousarray(h))
        file_t.append(np.ascontiguousarray(t))

    masks = [np.zeros(int((d or 0) * fps) + 1, dtype=bool) for d in durations]
    min_frames = int(m.min_length_sec * fps)
    exhausted: set[int] = set()
    segments: list[Segment] = []

    for _iteration in range(_MAX_ITERATIONS):
        _check_cancel(cancel)
        best_pivot = None
        best_unmasked = 0
        for i in range(n):
            if i in exhausted:
                continue
            unmasked = len(masks[i]) - int(np.count_nonzero(masks[i]))
            if unmasked > best_unmasked:
                best_unmasked = unmasked
                best_pivot = i
        if best_pivot is None:
            break

        ph, pt = unmasked_hashes(file_h[best_pivot], file_t[best_pivot], masks[best_pivot])
        if len(ph) < m.min_count:
            exhausted.add(best_pivot)
            continue
        pivot_index = mt.HashIndex(ph, pt)
        # A mask only changes after its own file's propagation below, so one
        # filtered view per file serves both the search and propagation.
        filtered = [
            (ph, pt) if i == best_pivot else unmasked_hashes(file_h[i], file_t[i], masks[i])
            for i in range(n)
        ]

        best = None
        for c in range(n):
            if c == best_pivot:
                continue
            ch, ct = filtered[c]
            if len(ch) == 0:
                continue
            t_piv, deltas = mt.offset_pairs(pivot_index, ch, ct)
            if t_piv.size == 0:
                continue
            peak_deltas, _ = mt.histogram_peaks(deltas, m.min_count)
            if peak_deltas.size == 0:
                continue
            lookup = mt.DeltaLookup(t_piv, deltas)
            for delta in peak_deltas.tolist():
                frames = lookup.frames(delta, m.delta_tolerance)
                runs = mt.contiguous_runs(frames, m.gap_frames * 3)
                for start, end, run_count in mt.filter_runs_by_length(
                    runs, min_frames, hcfg.dt_min, hcfg.dt_max
                ):
                    if best is None or run_count > best[4]:
                        best = (c, delta, start, end, run_count)

        if best is None:
            exhausted.add(best_pivot)
            continue

        cand, _delta, seg_start, seg_end, _ = best
        seg_duration = (seg_end - seg_start) / fps
        _log(
            f"  Segment {len(segments) + 1}: "
            f"frames {seg_start}-{seg_end} ({seg_duration:.1f}s) "
            f"pivot={best_pivot + 1}, matched with file {cand + 1}"
        )

        in_segment = (pt >= seg_start) & (pt <= seg_end)
        canonical = mt.HashIndex(ph[in_segment], pt[in_segment] - seg_start)

        matches = []
        for i in range(n):
            fh, ft = filtered[i]
            result = mt.propagate_segment(
                canonical, fh, ft,
                min_count=m.min_count,
                delta_tolerance=m.delta_tolerance,
                threshold=m.propagation_threshold,
                gap_frames=m.gap_frames,
                dt_min=hcfg.dt_min,
                dt_max=hcfg.dt_max,
            )
            if result is None:
                continue
            m_start, m_end, score = result
            m_start_sec = m_start / fps
            m_end_sec = m_end / fps
            m_end_sec = min(m_end_sec, m_start_sec + seg_duration)
            m_end = min(m_end, m_start + seg_end - seg_start)
            matches.append(SegmentMatch(i, m_start, m_end, m_start_sec, m_end_sec, score))
            mask_range(masks[i], m_start, m_end)

        segments.append(Segment(best_pivot, cand, seg_start, seg_end, seg_duration, tuple(matches)))

    _log(f"  Analysis complete: {len(segments)} segment(s) found.")
    return segments


# ---------------------------------------------------------------------------
# Keep ranges
# ---------------------------------------------------------------------------

def _secs_to_mmss(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def merge_silence_gaps(
    file_skip_blocks: dict[int, list[tuple[float, float]]],
    position_tolerance_sec: float = 5.0,
    max_gap_duration_sec: float = 30.0,
    min_file_ratio: float = 0.5,
) -> dict[int, list[tuple[float, float]]]:
    """Bridge silence gaps between skip blocks when they appear at a similar
    position in more than ``min_file_ratio`` of files. Original code."""
    all_gaps: list[tuple[int, int, float, float, float]] = []
    for fid, blocks in file_skip_blocks.items():
        for i in range(len(blocks) - 1):
            gap_start = blocks[i][1]
            gap_end = blocks[i + 1][0]
            if gap_end - gap_start <= max_gap_duration_sec:
                all_gaps.append((fid, i, gap_start, gap_end, (gap_start + gap_end) / 2.0))

    if not all_gaps:
        return file_skip_blocks

    min_file_count = max(2, int(len(file_skip_blocks) * min_file_ratio))

    all_gaps.sort(key=lambda g: g[4])
    clusters: list[list[tuple[int, int, float, float, float]]] = []
    current = [all_gaps[0]]
    for gap in all_gaps[1:]:
        if gap[4] - current[0][4] <= position_tolerance_sec * 2:
            if gap[0] not in {g[0] for g in current}:
                current.append(gap)
            else:
                clusters.append(current)
                current = [gap]
        else:
            clusters.append(current)
            current = [gap]
    clusters.append(current)

    gaps_to_bridge: set[tuple[int, int]] = set()
    for cluster in clusters:
        if len({g[0] for g in cluster}) >= min_file_count:
            for fid, gap_idx, _, _, _ in cluster:
                gaps_to_bridge.add((fid, gap_idx))

    if not gaps_to_bridge:
        return file_skip_blocks

    result: dict[int, list[tuple[float, float]]] = {}
    for fid, blocks in file_skip_blocks.items():
        new_blocks: list[tuple[float, float]] = []
        i = 0
        while i < len(blocks):
            start, end = blocks[i]
            while i < len(blocks) - 1 and (fid, i) in gaps_to_bridge:
                end = max(end, blocks[i + 1][1])
                i += 1
            new_blocks.append((start, end))
            i += 1
        result[fid] = new_blocks
    return result


def compute_keep_ranges(
    segments: Sequence[Segment],
    names: Sequence[str],
    durations: Sequence[float],
    merge_repeating_silences: bool = False,
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Invert every file's merged skip blocks into keep ranges (MM:SS).

    Files with no match, or no gap of at least MIN_GAP_SEC, are absent from
    the result. The dict's order is the original's: files in order of first
    appearance, walking segments in discovery order and each segment's
    matches by descending score (ties in file order -- what the original's
    ``ORDER BY score DESC`` returned).
    """
    file_matches: dict[int, list[SegmentMatch]] = {}
    for seg in segments:
        for match in sorted(seg.matches, key=lambda mm: -mm.score):
            file_matches.setdefault(match.file, []).append(match)

    file_skip_blocks: dict[int, list[tuple[float, float]]] = {}
    for fid, matches in file_matches.items():
        if not 0 <= fid < len(names):
            continue
        matches.sort(key=lambda mm: mm.start_sec)
        merged: list[tuple[float, float]] = []
        for mm in matches:
            s, e = mm.start_sec, mm.end_sec
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        file_skip_blocks[fid] = merged

    if merge_repeating_silences and len(file_skip_blocks) >= 2:
        file_skip_blocks = merge_silence_gaps(file_skip_blocks)

    results: dict[str, list[tuple[str | None, str | None]]] = {}
    for fid, merged in file_skip_blocks.items():
        duration = durations[fid] or 0.0
        keep_ranges: list[tuple[str | None, str | None]] = []
        cursor = 0.0
        for skip_start, skip_end in merged:
            if skip_start - cursor >= MIN_GAP_SEC:
                start_str = _secs_to_mmss(cursor) if cursor > 0 else None
                keep_ranges.append((start_str, _secs_to_mmss(skip_start)))
            cursor = skip_end
        if duration > 0 and (duration - cursor) >= MIN_GAP_SEC:
            start_str = _secs_to_mmss(cursor) if cursor > 0 else None
            keep_ranges.append((start_str, None))
        if keep_ranges:
            results[names[fid]] = keep_ranges
    return results


# ---------------------------------------------------------------------------
# Blocks (evidence: raw matched segments per file, kind, count, score)
# ---------------------------------------------------------------------------

# The band, at each end of a file, within which a matched segment counts as
# "intro"/"outro" rather than a plain repeat. Picked, not measured: 5% of a
# typical 20-40 min episode is 1-2 minutes -- generous enough for a cold-open
# or a trailing sponsor card -- while the 5.0 s floor keeps that same margin
# meaningful on short clips instead of shrinking to nothing.
_EDGE_MIN_SEC = 5.0
_EDGE_FRACTION = 0.05


def _block_kind(start_sec: float, end_sec: float, duration: float) -> str:
    """"intro" if the match starts at or before the edge band, else "outro"
    if it ends at or after the edge band from the end, else "repeat". Both
    comparisons are inclusive of the edge itself (<=, >=): a match landing
    exactly on the boundary counts as intro/outro, not as a plain repeat."""
    edge = max(_EDGE_MIN_SEC, _EDGE_FRACTION * duration)
    if start_sec <= edge:
        return "intro"
    if end_sec >= duration - edge:
        return "outro"
    return "repeat"


def compute_blocks(
    segments: Sequence[Segment],
    names: Sequence[str],
    durations: Sequence[float],
) -> dict[str, list[Block]]:
    """One Block per SegmentMatch that belongs to a file, keyed by filename.

    Deliberately NOT the merged skip spans compute_keep_ranges() derives
    from the same segments: if two matches in the same file overlap (across
    different discovered segments), both are kept as separate Blocks here,
    unmerged. Each file's list is sorted by start_sec. ``matched_files`` is
    the number of distinct files among that segment's own matches (not
    ``len(segment.matches)`` -- a segment is not guaranteed to have exactly
    one match per file); ``score`` is this file's own SegmentMatch.score.
    Files with no match are absent from the result, matching
    compute_keep_ranges()'s own convention.
    """
    blocks: dict[str, list[Block]] = {}
    for seg in segments:
        matched_files = len({mm.file for mm in seg.matches})
        for mm in seg.matches:
            if not 0 <= mm.file < len(names):
                continue
            duration = durations[mm.file] or 0.0
            kind = _block_kind(mm.start_sec, mm.end_sec, duration)
            blocks.setdefault(names[mm.file], []).append(
                Block(mm.start_sec, mm.end_sec, kind, matched_files, mm.score)
            )
    for file_blocks in blocks.values():
        file_blocks.sort(key=lambda b: b.start_sec)
    return blocks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyse_detailed(
    files: Sequence[FileEntry],
    cfg: RangesConfig,
    progress: ProgressFn | None = None,
    *,
    cache_dir: str | None = None,
    workers: int = DEFAULT_WORKERS,
    cancel: CancelFn | None = None,
) -> RangesAnalysis:
    """Keep ranges, plus the evidence analyse() drops: each file's raw
    matched blocks (kind/matched_files/score, see compute_blocks()) and
    every analysed file's duration.

    Same parameters, same cache behaviour, same exceptions as analyse():
    AnalysisCancelled when ``cancel()`` returns True, decode errors (no
    audio stream, ffmpeg failure) propagate uncaught. analyse()'s own
    return is exactly this call's ``.keep``.
    """
    def emit(event: ProgressEvent) -> None:
        if progress is not None:
            progress(event)

    emit(ProgressEvent("phase", "Fingerprinting"))
    fingerprints, durations = ingest(
        files, cfg, cache_dir=cache_dir, workers=workers, progress=progress, cancel=cancel,
    )
    _check_cancel(cancel)

    emit(ProgressEvent("phase", "Analyzing"))
    segments = discover_segments(
        fingerprints, durations, cfg,
        log=lambda msg: emit(ProgressEvent("log", msg)), cancel=cancel,
    )
    _check_cancel(cancel)

    emit(ProgressEvent("phase", "Computing time ranges"))
    names = [f.name for f in files]
    keep = compute_keep_ranges(segments, names, durations, cfg.merge_repeating_silences)
    blocks = compute_blocks(segments, names, durations)
    durations_by_name = dict(zip(names, durations))

    return RangesAnalysis(keep=keep, blocks=blocks, durations=durations_by_name)


def analyse(
    files: Sequence[FileEntry],
    cfg: RangesConfig,
    progress: ProgressFn | None = None,
    *,
    cache_dir: str | None = None,
    workers: int = DEFAULT_WORKERS,
    cancel: CancelFn | None = None,
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Keep ranges per filename for a folder of episodes.

    ``cache_dir`` is where fingerprints are cached (normally
    ``default_cache_dir(project)``); None disables caching and writes
    nothing. Raises AnalysisCancelled when ``cancel()`` returns True, and
    lets decode errors (no audio stream, ffmpeg failure) propagate.
    """
    return analyse_detailed(
        files, cfg, progress, cache_dir=cache_dir, workers=workers, cancel=cancel,
    ).keep
