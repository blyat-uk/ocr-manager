"""Project JSON store: v2 `.ocr.json` read/write, v1 migration dispatch,
and reconciliation against the video files actually on disk.

Reading `.ocr.json` (load_project)
    - no file: a fresh project;
    - "version" missing or 1: migrated from v1 (migrate_v1);
    - "version" 2: from_json;
    - any other version (an integer other than 1 or 2, or not an integer,
      e.g. "2", 2.0 or true): UnsupportedProjectVersion is raised and the
      file is left untouched; save_project never writes over such a file;
    - unreadable or invalid JSON, JSON that is not an object, or a file the
      reader cannot convert (a bad enum value, a missing required key, a
      section of the wrong type): logged as a warning, renamed to
      `.ocr.json.corrupt-<unix time>`, and a fresh project is started.

Evidence (spec SS8.3: caches live in `.ocr-cache/`)
    In memory, FileEntry.evidence is the dict core.jobs.apply and AutoPilot
    use. On disk, `.ocr.json` holds no evidence: each file's evidence is its
    own cache file, `<project>/.ocr-cache/evidence/<sha256 hex of the file
    name's bytes, os.fsencode>.json` (evidence_path), holding compact UTF-8
    JSON {"version": 1, "name": <file name>, "evidence": {...}}. The name is
    hashed, not quoted, so every name gives a short filesystem-safe file
    name; the name inside the file ties it back to its entry.
    - save_project writes `.ocr.json` first, then a file's evidence only
      when its serialised form differs from what was last loaded or saved
      for it (a digest per file name, kept on the Project:
      Project.evidence_digests), or when its cache file is gone. A file with
      empty evidence has no evidence file. Evidence files of files no longer
      in the project (or whose evidence was emptied) are deleted; other files
      in the directory are left alone. The cache never stops the user's
      values being saved: an evidence write or delete that fails (OSError)
      is logged, the file's digest stays unset, and the next save retries.
    - load_project reads the evidence file of every entry. A corrupt one
      (not JSON, not an object, another file's name, no evidence object)
      gives empty evidence and a warning; so does a missing one when the
      entry has flags (a detection result was applied, so evidence was
      stored).
    - The one piece of evidence the review state depends on is not left to
      the cache: evidence["brightness"]["value_crop_box"], the crop a
      detected or hinted brightness value was measured on
      (core.jobs.apply.brightness_is_stale). Every file entry of `.ocr.json`
      carries it as "brightness_crop_box" ([x, y, w, h] or null), and on load
      it is authoritative: it replaces (or, when null, removes) the value
      the evidence cache or inline evidence holds, creating
      evidence["brightness"] = {"value_crop_box": ...} when the cache is
      gone. A v2 file entry without the key (written before it existed)
      keeps whatever its evidence says. Everything else about a file's
      values, sources and review is in `.ocr.json` alone.
    - A v2 `.ocr.json` written before evidence moved out (evidence inline in
      each file entry) still loads, and the next save moves the evidence out.
      An entry's inline evidence wins over an evidence file: it belongs to
      the values in the same file.

Writing
    Every file is written atomically: to `<name>.tmp` in the same directory,
    flushed and fsynced, then os.replace()d onto `<name>`. Everything written
    is JSON in UTF-8. A file name that is not valid UTF-8 (a lone surrogate
    from os.fsdecode) is written as its JSON escape (\\udcXX), which reads
    back as the same name.
"""
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from core.project.migrate import migrate_v1
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRange,
    TimeRanges,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mkv", ".mp4")
CONFIG_FILENAME = ".ocr.json"
SUPPORTED_VERSIONS = (1, 2)       # 1 (or no version): migrated; 2: current
CACHE_DIRNAME = ".ocr-cache"      # the project's cache directory (same as core.detect.ranges.pipeline's)
EVIDENCE_DIRNAME = "evidence"
EVIDENCE_FORMAT_VERSION = 1
_EVIDENCE_FILE_RE = re.compile(r"^[0-9a-f]{64}\.json$")


class UnsupportedProjectVersion(Exception):
    """`.ocr.json` has a version this app cannot read (newer, or not an
    integer). The file is left untouched."""

    def __init__(self, path: str, version):
        super().__init__(f"{path} has unsupported project version {version!r}")
        self.path = path
        self.version = version


class _CorruptProject(Exception):
    """The file parsed as JSON but is not a project this reader can convert."""


# Errors a malformed but parseable project file raises while it is converted.
_CONVERSION_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def list_video_files(project_dir: str) -> list[str]:
    """Sorted video file names in `project_dir` -- same rule as the v1 app's
    core/pipeline.py get_video_files (glob on VIDEO_EXTENSIONS,
    case-sensitive).
    """
    directory = Path(project_dir)
    names = [
        f.name
        for ext in VIDEO_EXTENSIONS
        for f in directory.glob(f"*{ext}")
    ]
    return sorted(names)


def load_project(project_dir: str) -> Project:
    directory = Path(project_dir)
    config_path = directory / CONFIG_FILENAME
    video_names = list_video_files(project_dir)

    project: Project
    brightness_records: dict[str, list[int] | None] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise _CorruptProject(f"top level is {type(data).__name__}, not an object")
            version = _config_version(data, config_path)
            try:
                if version == 2:
                    project, brightness_records = _from_json(data, str(directory))
                else:
                    project = migrate_v1(data, str(directory), video_names)
            except _CONVERSION_ERRORS as exc:
                raise _CorruptProject(f"{type(exc).__name__}: {exc}") from exc
        except (OSError, ValueError, _CorruptProject) as exc:      # JSONDecodeError/UnicodeDecodeError are ValueErrors
            logger.warning(
                "Corrupt %s in %s (%s) -- renaming and starting a fresh project",
                CONFIG_FILENAME, project_dir, exc,
            )
            corrupt_path = directory / f"{CONFIG_FILENAME}.corrupt-{int(time.time())}"
            try:
                config_path.rename(corrupt_path)
            except OSError:
                logger.warning("Could not rename corrupt config %s", config_path)
            project = Project(path=str(directory), folder=FolderSettings(), files={})
    else:
        project = Project(path=str(directory), folder=FolderSettings(), files={})

    reconcile_files(project, video_names)
    _load_evidence(project)
    _apply_brightness_records(project, brightness_records)
    return project


def evidence_path(project_dir: str, name: str) -> Path:
    """The evidence cache file of the video file `name` in `project_dir`."""
    digest = hashlib.sha256(os.fsencode(name)).hexdigest()
    return Path(project_dir) / CACHE_DIRNAME / EVIDENCE_DIRNAME / f"{digest}.json"


def _evidence_text(name: str, evidence: dict) -> str:
    return json.dumps({"version": EVIDENCE_FORMAT_VERSION, "name": name, "evidence": evidence},
                      separators=(",", ":"), ensure_ascii=False)


def _encode(text: str) -> bytes:
    """UTF-8 bytes of JSON text; a lone surrogate (a name that is not valid
    UTF-8) becomes its JSON escape."""
    return text.encode("utf-8", "backslashreplace")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _apply_brightness_records(project: Project, records: dict[str, list[int] | None]) -> None:
    """Make `.ocr.json`'s brightness_crop_box authoritative over the evidence
    (see the module docstring). `records` holds only entries that had the key."""
    for name, box in records.items():
        entry = project.files.get(name)
        if entry is None:
            continue
        brightness = entry.evidence.get("brightness")
        if box is not None:
            if not isinstance(brightness, dict):
                brightness = entry.evidence["brightness"] = {}
            brightness["value_crop_box"] = list(box)
        elif isinstance(brightness, dict) and "value_crop_box" in brightness:
            del brightness["value_crop_box"]
            if not brightness:
                del entry.evidence["brightness"]


def _brightness_record(entry: FileEntry) -> list[int] | None:
    """The entry's evidence["brightness"]["value_crop_box"], as `.ocr.json` stores it."""
    brightness = entry.evidence.get("brightness")
    box = brightness.get("value_crop_box") if isinstance(brightness, dict) else None
    return None if box is None else [int(v) for v in box]


def _parse_brightness_record(value) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError(f"brightness_crop_box must be null or four integers, not {value!r}")
    return list(value)


def _load_evidence(project: Project) -> None:
    """Read every entry's evidence file (see the module docstring), and seed
    project.evidence_digests with what was read."""
    missing = []
    for name, entry in project.files.items():
        if entry.evidence:                           # inline, from a v2 file written before the cache
            continue
        path = evidence_path(project.path, name)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            if entry.flags:
                missing.append(name)
            continue
        except OSError as exc:
            logger.warning("Could not read the evidence of %s (%s): evidence is empty", name, exc)
            continue
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("name") != name or not isinstance(data.get("evidence"), dict):
                raise ValueError("not an evidence file for this name")
        except ValueError as exc:                    # JSONDecodeError and UnicodeDecodeError included
            logger.warning("Corrupt evidence cache %s for %s (%s): evidence is empty", path, name, exc)
            continue
        entry.evidence = data["evidence"]
        project.evidence_digests[name] = _digest(raw)
    if missing:
        logger.warning("Evidence cache missing for %d file(s) with detection results (%s): evidence is empty",
                       len(missing), ", ".join(missing))


def _save_evidence(project: Project) -> None:
    """Write changed evidence files and delete stale ones (see the module
    docstring). Never raises OSError: failures are logged and retried by the
    next save."""
    directory = Path(project.path) / CACHE_DIRNAME / EVIDENCE_DIRNAME
    try:
        existing = {item.name for item in os.scandir(directory) if _EVIDENCE_FILE_RE.match(item.name)}
    except FileNotFoundError:
        existing = set()
    except OSError as exc:
        logger.warning("Could not list the evidence cache %s (%s)", directory, exc)
        existing = set()
    digests = project.evidence_digests
    kept: set[str] = set()
    unwritten: list[str] = []
    first_error: OSError | None = None
    for name, entry in project.files.items():
        if not entry.evidence:
            continue
        path = evidence_path(project.path, name)
        kept.add(path.name)
        text = _evidence_text(name, entry.evidence)
        digest = _digest(_encode(text))
        if digests.get(name) == digest and path.name in existing:
            continue
        digests.pop(name, None)                      # unknown until the write succeeds
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, text)
        except OSError as exc:
            unwritten.append(name)
            first_error = first_error or exc
            continue
        digests[name] = digest
    if unwritten:
        logger.warning("Could not write the evidence cache of %d file(s) (%s): %s; retried on the next save",
                       len(unwritten), ", ".join(unwritten), first_error)
    undeleted: list[str] = []
    for stale in sorted(existing - kept):
        try:
            os.remove(directory / stale)
        except FileNotFoundError:
            pass
        except OSError as exc:
            undeleted.append(stale)
            first_error = exc
    if undeleted:
        logger.warning("Could not delete %d stale evidence cache file(s) in %s: %s; retried on the next save",
                       len(undeleted), directory, first_error)
    for name in [n for n in digests if n not in project.files or not project.files[n].evidence]:
        del digests[name]


def _config_version(data: dict, config_path: Path) -> int:
    """1 (no version, or 1) or 2; UnsupportedProjectVersion otherwise."""
    version = data.get("version", 1)
    if type(version) is not int or version not in SUPPORTED_VERSIONS:   # bool, float, str are not versions
        raise UnsupportedProjectVersion(str(config_path), version)
    return version


def _refuse_unsupported_existing(config_path: Path) -> None:
    """Raise UnsupportedProjectVersion when `config_path` holds a project of a
    version this app cannot read; anything else (no file, unreadable,
    corrupt, v1, v2) may be written over."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        _config_version(data, config_path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write JSON `text` (UTF-8, see _encode) to `path` atomically:
    `<path>.tmp` in the same directory, flushed and fsynced, then os.replace()
    onto `path`. The temporary file is removed when anything fails."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(_encode(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_project(project: Project) -> None:
    directory = Path(project.path)
    config_path = directory / CONFIG_FILENAME
    backup_path = directory / f"{CONFIG_FILENAME}.v1.bak"

    _refuse_unsupported_existing(config_path)

    if project.migrated_from_v1:
        if not backup_path.exists() and config_path.exists():
            shutil.copy2(config_path, backup_path)
        project.migrated_from_v1 = False

    _atomic_write_text(config_path,
                       json.dumps(to_json(project, include_evidence=False), ensure_ascii=False, indent=2))
    _save_evidence(project)


def reconcile_files(project: Project, video_names: list[str]) -> tuple[list[str], list[str]]:
    existing = set(project.files.keys())
    incoming = set(video_names)

    added = sorted(incoming - existing)
    removed = sorted(existing - incoming)

    for name in removed:
        del project.files[name]
    for name in added:
        project.files[name] = FileEntry(name=name)

    ordered = {name: project.files[name] for name in sorted(project.files.keys())}
    project.files.clear()
    project.files.update(ordered)

    return added, removed


def to_json(project: Project, *, include_evidence: bool = True) -> dict:
    """The v2 dict of `project`. `.ocr.json` is written with
    include_evidence=False: its file entries then have no "evidence" key
    (evidence lives in the evidence cache, see the module docstring). Every
    file entry carries "brightness_crop_box" either way."""
    folder = project.folder
    folder_dict = {
        "dialogue_enabled": folder.dialogue_enabled,
        "labels_enabled": folder.labels_enabled,
        "ocr_lang": folder.ocr_lang,
        "conf_threshold": folder.conf_threshold,
        "sim_threshold": folder.sim_threshold,
        "similar_image": folder.similar_image,
        "frames_to_skip": folder.frames_to_skip,
        "use_gpu": folder.use_gpu,
        "label_min_duration": folder.label_min_duration,
        "label_max_duration": folder.label_max_duration,
        "label_conf_threshold": folder.label_conf_threshold,
        "label_conf_threshold_min": folder.label_conf_threshold_min,
        "label_mask_crops": [list(c) for c in folder.label_mask_crops],
        "ocr_parallel": folder.ocr_parallel,
        "autopilot_enabled": folder.autopilot_enabled,
        "brightness_full_detect_files": folder.brightness_full_detect_files,
        "min_segment_length": folder.min_segment_length,
        "merge_repeating_silences": folder.merge_repeating_silences,
        "crop_width_fraction": folder.crop_width_fraction,
        "crop_vertical_padding": folder.crop_vertical_padding,
        "crop_min_height_fraction": folder.crop_min_height_fraction,
        "bottom_half_cutoff": folder.bottom_half_cutoff,
    }

    files_dict = {}
    for name, entry in project.files.items():
        files_dict[name] = {
            "crop": None if entry.crop is None else {
                "x": entry.crop.x,
                "y": entry.crop.y,
                "width": entry.crop.width,
                "height": entry.crop.height,
                "source": entry.crop.source.value,
            },
            "brightness": None if entry.brightness is None else {
                "value": entry.brightness.value,
                "source": entry.brightness.source.value,
            },
            "brightness_crop_box": _brightness_record(entry),   # authoritative for staleness on load
            "time_ranges": None if entry.time_ranges is None else {
                "ranges": [{"start": r.start, "end": r.end} for r in entry.time_ranges.ranges],
                "source": entry.time_ranges.source.value,
            },
            "media": {
                "width": entry.media.width,
                "height": entry.media.height,
                "duration": entry.media.duration,
                "fps": entry.media.fps,
            },
            "review": entry.review.value,
            "skipped": entry.skipped,
            "sample_time": entry.sample_time,
            "flags": dict(entry.flags),  # str -> str: a shallow copy is a full copy, values are scalar
        }
        if include_evidence:
            # str -> dict: a shallow copy would still alias the nested dicts
            files_dict[name]["evidence"] = copy.deepcopy(entry.evidence)

    return {
        "version": 2,
        "folder": folder_dict,
        "files": files_dict,
    }


def _object(value, what: str) -> dict:
    """`value` when it is a JSON object, {} when it is missing (None)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{what} is {type(value).__name__}, not an object")
    return value


def from_json(data: dict, project_dir: str) -> Project:
    """The Project a v2 dict describes. Raises KeyError, TypeError or
    ValueError (load_project's corrupt path) when a required key is missing,
    an enum value is unknown or a section is not an object. A file entry's
    "brightness_crop_box" is applied over its inline evidence."""
    project, brightness_records = _from_json(data, project_dir)
    _apply_brightness_records(project, brightness_records)
    return project


def _from_json(data: dict, project_dir: str) -> tuple[Project, dict[str, list[int] | None]]:
    """from_json without applying the brightness records: the Project (with
    only inline evidence) and the brightness_crop_box of every file entry
    that has the key."""
    defaults = FolderSettings()
    folder_data = _object(data.get("folder"), "folder")

    folder = FolderSettings(
        dialogue_enabled=folder_data.get("dialogue_enabled", defaults.dialogue_enabled),
        labels_enabled=folder_data.get("labels_enabled", defaults.labels_enabled),
        ocr_lang=folder_data.get("ocr_lang", defaults.ocr_lang),
        conf_threshold=folder_data.get("conf_threshold", defaults.conf_threshold),
        sim_threshold=folder_data.get("sim_threshold", defaults.sim_threshold),
        similar_image=folder_data.get("similar_image", defaults.similar_image),
        frames_to_skip=folder_data.get("frames_to_skip", defaults.frames_to_skip),
        use_gpu=folder_data.get("use_gpu", defaults.use_gpu),
        label_min_duration=folder_data.get("label_min_duration", defaults.label_min_duration),
        label_max_duration=folder_data.get("label_max_duration", defaults.label_max_duration),
        label_conf_threshold=folder_data.get("label_conf_threshold", defaults.label_conf_threshold),
        label_conf_threshold_min=folder_data.get(
            "label_conf_threshold_min", defaults.label_conf_threshold_min
        ),
        label_mask_crops=[tuple(c) for c in folder_data.get("label_mask_crops", [])],
        ocr_parallel=folder_data.get("ocr_parallel", defaults.ocr_parallel),
        autopilot_enabled=folder_data.get("autopilot_enabled", defaults.autopilot_enabled),
        brightness_full_detect_files=folder_data.get(
            "brightness_full_detect_files", defaults.brightness_full_detect_files
        ),
        min_segment_length=folder_data.get("min_segment_length", defaults.min_segment_length),
        merge_repeating_silences=folder_data.get(
            "merge_repeating_silences", defaults.merge_repeating_silences
        ),
        crop_width_fraction=folder_data.get("crop_width_fraction", defaults.crop_width_fraction),
        crop_vertical_padding=folder_data.get("crop_vertical_padding", defaults.crop_vertical_padding),
        crop_min_height_fraction=folder_data.get(
            "crop_min_height_fraction", defaults.crop_min_height_fraction
        ),
        bottom_half_cutoff=folder_data.get("bottom_half_cutoff", defaults.bottom_half_cutoff),
    )

    files: dict[str, FileEntry] = {}
    brightness_records: dict[str, list[int] | None] = {}
    for name, fd in _object(data.get("files"), "files").items():
        if not isinstance(fd, dict):
            raise TypeError(f"files[{name!r}] is {type(fd).__name__}, not an object")
        if "brightness_crop_box" in fd:
            brightness_records[name] = _parse_brightness_record(fd["brightness_crop_box"])
        crop_d = fd.get("crop")
        crop = None
        if crop_d:
            crop = Crop(
                x=crop_d["x"], y=crop_d["y"], width=crop_d["width"], height=crop_d["height"],
                source=Source(crop_d["source"]),
            )

        brightness_d = fd.get("brightness")
        brightness = None
        if brightness_d:
            brightness = Brightness(value=brightness_d["value"], source=Source(brightness_d["source"]))

        tr_d = fd.get("time_ranges")
        time_ranges = None
        if tr_d:
            ranges = [TimeRange(start=r.get("start"), end=r.get("end")) for r in tr_d.get("ranges", [])]
            time_ranges = TimeRanges(ranges=ranges, source=Source(tr_d["source"]))

        media_d = fd.get("media") or {}
        media = Media(
            width=media_d.get("width", 0),
            height=media_d.get("height", 0),
            duration=media_d.get("duration", 0.0),
            fps=media_d.get("fps", 0.0),
        )

        files[name] = FileEntry(
            name=name,
            crop=crop,
            brightness=brightness,
            time_ranges=time_ranges,
            media=media,
            review=ReviewState(fd.get("review", ReviewState.PENDING.value)),
            skipped=fd.get("skipped", False),
            sample_time=fd.get("sample_time"),
            flags=dict(fd.get("flags") or {}),
            evidence=copy.deepcopy(fd.get("evidence") or {}),
        )

    return Project(path=project_dir, folder=folder, files=files, migrated_from_v1=False), brightness_records
